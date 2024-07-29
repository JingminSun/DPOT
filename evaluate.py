import sys
import os
# sys.path.append(['.','./../'])
os.environ['OMP_NUM_THREADS'] = '16'
import matplotlib.pyplot as plt
import json
import time
import argparse
import torch
import numpy as np

from torch.optim.lr_scheduler import OneCycleLR, StepLR, LambdaLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.tensorboard import SummaryWriter
from utils.optimizer import Adam, Lamb
from utils.utilities import count_parameters, get_grid, load_model_from_checkpoint
from utils.criterion import SimpleLpLoss
from utils.griddataset import MixedTemporalDataset
from models.fno import FNO2d
from models.dpot import DPOTNet
import wandb

# torch.manual_seed(0)
# np.random.seed(0)



################################################################
# configs
################################################################


parser = argparse.ArgumentParser(description='Training or pretraining for the same data type')

### currently no influence
parser.add_argument('--model', type=str, default='DPOT')
parser.add_argument('--dataset',type=str, default='ns2d')

parser.add_argument('--train_paths', nargs='+', type=str, default=[
    'ns2d_pdb_M1_eta1e-1_zeta1e-1',
    'ns2d_pdb_M1_eta1e-2_zeta1e-2',
    'ns2d_pdb_M1e-1_eta1e-1_zeta1e-1',
    'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2',
    'ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_turb_128',
    'ns2d_pdb_M1_eta1e-8_zeta1e-8_turb_128',
    'ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_rand_128',
    'ns2d_pdb_M1_eta1e-8_zeta1e-8_rand_128',
    'ns2d_pdb_incom',
    'swe_pdb',
    'ns2d_cond_pda',
    'ns2d_pda',
    'cfdbench',
])
parser.add_argument('--test_paths', nargs='+', type=str,
                    default=[
                        'ns2d_pdb_M1_eta1e-1_zeta1e-1', 'ns2d_pdb_M1_eta1e-2_zeta1e-2',
                             'ns2d_pdb_M1e-1_eta1e-1_zeta1e-1', 'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2',
                             'ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_turb_128', 'ns2d_pdb_M1_eta1e-8_zeta1e-8_turb_128',
                             'ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_rand_128',
                             'ns2d_pdb_M1_eta1e-8_zeta1e-8_rand_128',
                             'ns2d_pdb_incom',
                        'swe_pdb',
                        'ns2d_cond_pda',
                        'ns2d_pda',
                        'cfdbench',
                    ])
parser.add_argument('--resume_path',type=str, default='logs_pretrain/DPOT_new_0726_08_41_19M_13_53346/model.pth')
parser.add_argument('--ntrain_list', nargs='+', type=int, default=[
    8000,
    8000,
    8000,
    8000,
    800,
    800,
    800,
    800,
    876,
    800,
    2496,
    5200,
    8774
])
parser.add_argument('--data_weights',nargs='+',type=int, default=[1])
parser.add_argument('--use_writer', action='store_true',default=False)

parser.add_argument('--res', type=int, default=128)
parser.add_argument('--noise_scale',type=float, default=0)
# parser.add_argument('--n_channels',type=int,default=-1)


### shared params
parser.add_argument('--width', type=int, default=1024)
parser.add_argument('--n_layers',type=int, default=12)
parser.add_argument('--act',type=str, default='gelu')

### GNOT params
parser.add_argument('--max_nodes',type=int, default=-1)

### FNO params
parser.add_argument('--modes', type=int, default=32)
parser.add_argument('--use_ln',type=int, default=0)
parser.add_argument('--normalize',type=int, default=0)


### AFNO
parser.add_argument('--patch_size',type=int, default=8)
parser.add_argument('--n_blocks',type=int, default=8)
parser.add_argument('--mlp_ratio',type=int, default=4)
parser.add_argument('--out_layer_dim', type=int, default=32)


parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--opt',type=str, default='adam', choices=['adam','lamb'])
parser.add_argument('--beta1',type=float,default=0.9)
parser.add_argument('--beta2',type=float,default=0.9)
parser.add_argument('--lr_method',type=str, default='cycle')
parser.add_argument('--grad_clip',type=float, default=10000.0)
parser.add_argument('--step_size', type=int, default=100)
parser.add_argument('--step_gamma', type=float, default=0.5)
parser.add_argument('--warmup_epochs',type=int, default=200)
parser.add_argument('--sub', type=int, default=1)
parser.add_argument('--T_in', type=int, default=10)
parser.add_argument('--T_ar', type=int, default=1)
parser.add_argument('--T_bundle', type=int, default=1)
parser.add_argument('--gpu', type=str, default="3")
parser.add_argument('--comment',type=str, default="")
parser.add_argument('--log_path',type=str,default='')


parser.add_argument('--n_channels',type=int, default=4)
parser.add_argument('--n_class',type=int,default=13)

args = parser.parse_args()


device = torch.device("cuda:{}".format(args.gpu))

print(f"Current working directory: {os.getcwd()}")




################################################################
# load data and dataloader
################################################################
train_paths = args.train_paths
test_paths = args.test_paths
args.data_weights = [1] * len(args.train_paths) if len(args.data_weights) == 1 else args.data_weights
print('args',args)


train_dataset = MixedTemporalDataset(args.train_paths, args.ntrain_list, res=args.res, t_in = args.T_in, t_ar = args.T_ar, normalize=False,train=True, valid=False, data_weights=args.data_weights, n_channels=args.n_channels)
# test_datasets = [MixedTemporalDataset(test_path, [args.ntest_list[i]], res=args.res, n_channels = train_dataset.n_channels,t_in = args.T_in, t_ar=-1, normalize=False, train=False, valid=False) for i, test_path in enumerate(test_paths)]
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
test_datasets = [
    MixedTemporalDataset(test_path, res=args.res, n_channels=train_dataset.n_channels, t_in=args.T_in, t_ar=-1,
                         normalize=False, train=False, valid=False) for test_path in test_paths]

test_loaders = [torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8) for test_dataset in test_datasets]
ntrain, ntests = len(train_dataset), [len(test_dataset) for test_dataset in test_datasets]
print('Train num {} test num {}'.format(train_dataset.n_sizes, ntests))
################################################################
# load model
################################################################
if args.model == "FNO":
    model = FNO2d(args.modes, args.modes, args.width, img_size = args.res, patch_size=args.patch_size, in_timesteps = args.T_in, out_timesteps=1,normalize=args.normalize,n_layers = args.n_layers,use_ln = args.use_ln, n_channels=train_dataset.n_channels, n_cls=len(args.train_paths)).to(device)
elif args.model == 'DPOT':
    model = DPOTNet(img_size=args.res, patch_size=args.patch_size, in_channels=train_dataset.n_channels, in_timesteps = args.T_in, out_timesteps = args.T_bundle, out_channels=train_dataset.n_channels, normalize=args.normalize, embed_dim=args.width, modes=args.modes, depth=args.n_layers, n_blocks = args.n_blocks, mlp_ratio=args.mlp_ratio, out_layer_dim=args.out_layer_dim, act=args.act, n_cls=args.n_class).to(device)
else:
    raise NotImplementedError

if args.resume_path:
    print('Loading models and fine tune from {}'.format(args.resume_path))
    args.resume_path = args.resume_path

    load_model_from_checkpoint(model, torch.load(args.resume_path,map_location='cuda:{}'.format(args.gpu))['model'])

#### set optimizer
if args.opt == 'lamb':
    optimizer = Lamb(model.parameters(), lr=args.lr, betas = (args.beta1, args.beta2), adam=True, debias=False,weight_decay=1e-4)
else:
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=1e-6)


if args.lr_method == 'cycle':
    print('Using cycle learning rate schedule')
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, div_factor=1e4, pct_start=(args.warmup_epochs / args.epochs), final_div_factor=1e4, steps_per_epoch=len(train_loader), epochs=args.epochs)
elif args.lr_method == 'step':
    print('Using step learning rate schedule')
    scheduler = StepLR(optimizer, step_size=args.step_size * len(train_loader), gamma=args.step_gamma)
elif args.lr_method == 'warmup':
    print('Using warmup learning rate schedule')
    scheduler = LambdaLR(optimizer, lambda steps: min((steps + 1) / (args.warmup_epochs * len(train_loader)), np.power(args.warmup_epochs * len(train_loader) / float(steps + 1), 0.5)))
elif args.lr_method == 'linear':
    print('Using warmup learning rate schedule')
    scheduler = LambdaLR(optimizer, lambda steps: (1 - steps / (args.epochs * len(train_loader))))
elif args.lr_method == 'restart':
    print('Using cos anneal restart')
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader) * args.lr_step_size, eta_min=0.)
elif args.lr_method == 'cyclic':
    scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=args.lr_step_size * len(train_loader),mode='triangular2', cycle_momentum=False)
else:
    raise NotImplementedError

comment = args.comment + '_{}_{}'.format(len(train_paths), ntrain)
log_path = 'logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('logs',args.log_path + comment)
model_path = log_path + '/model.pth'
if args.use_writer:
    writer = SummaryWriter(log_dir=log_path)
    fp = open(log_path + '/logs.txt', 'w+',buffering=1)
    json.dump(vars(args), open(log_path + '/params.json', 'w'),indent=4)
    sys.stdout = fp

else:
    writer = None
print(model)
count_parameters(model)

if args.use_writer:
    wandb.init(
        project="dpot",
        resume="allow",
        id=wandb.util.generate_id(),
        name="eval_0722run_step10",
        entity="schaefferlab1",
        notes="",
    )

    wandb.config.update(args, allow_val_change=True)
    wandb.log({"id": "evaluation"})

################################################################
# Main function for pretraining
################################################################
myloss = SimpleLpLoss(size_average=False)
clsloss = torch.nn.CrossEntropyLoss(reduction='sum')


def get_error(residuals,tar):


    t = residuals.shape[1]

    error = torch.sqrt((residuals **2).flatten(2).sum(2))
    scale = 1e-8 + torch.sqrt((tar ** 2).flatten(2).sum(2))

    error = torch.div(error,scale).sum()/t
    return error


test_l2_fulls, test_l2_steps, time_test, total_steps = [], [], 0., 0
model.eval()
total_pdb=0
step_pdb =0
num_samples =0
with torch.no_grad():
    for id, test_loader in enumerate(test_loaders):
        test_name = args.test_paths[id]
        print(test_name)


        test_l2_full, test_l2_step = 0, 0
        nn = 0
        for xx, yy, msk, _ in test_loader:
            loss = 0
            error = 0
            xx = xx.to(device)
            yy = yy.to(device)
            msk = msk.to(device)
            nn += xx.shape[0]


            for t in range(0, yy.shape[-2], args.T_bundle):
                y = yy[..., t:t + args.T_bundle, :]

                time_i = time.time()
                im, _ = model(xx)
                time_test += time.time() - time_i

                loss += myloss(im, y, mask=msk)

                y_reshape = y.permute(0,3,4,1, 2) # (bs,h,w,t,c) to (bs,t,c,h,w)
                im_reshape = im.permute(0,3,4, 1, 2)
                error += get_error(im_reshape-y_reshape,y_reshape)
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -2)

                xx = torch.cat((xx[..., args.T_bundle:,:], im), dim=-2)
                total_steps += xx.shape[0]

            test_l2_step += error/ (yy.shape[-2] / args.T_bundle) # sum step error/5
            # test_l2_step += loss/ (yy.shape[-2] / args.T_bundle)
            # print(error/ (yy.shape[-2] / args.T_bundle), loss.item()/ (yy.shape[-2] / args.T_bundle))
            pred_reshape = pred.permute(0, 3, 1, 2,4)
            yy_reshape = yy.permute(0, 3, 1, 2,4)
            # print(get_error(pred_reshape-yy_reshape,yy_reshape),myloss(pred, yy, mask=msk) )
            test_l2_full += get_error(pred_reshape-yy_reshape,yy_reshape)
            # test_l2_full +=myloss(pred, yy, mask=msk)

        test_l2_step_avg, test_l2_full_avg = test_l2_step / ntests[id] , test_l2_full / ntests[id]
        test_l2_steps.append(test_l2_step_avg.item())
        test_l2_fulls.append(test_l2_full_avg.item())
        if "ns2d_pdb_M1" in test_name:
            total_pdb += test_l2_full
            step_pdb += test_l2_step
            num_samples += ntests[id]

        # Select a sample from the predictions and ground truth
        sample_idx = 0  # You can choose any index
        for jj in range(pred.shape[-1]):
            sample_pred = pred_reshape[sample_idx:sample_idx+1, :, :, :, :].cpu().numpy()
            sample_yy = yy_reshape[sample_idx:sample_idx+1, :, :, :, :].cpu().numpy()

            # Calculate error for the sample
            sample_error = get_error(torch.tensor(sample_pred-sample_yy), torch.tensor(sample_yy)).item()

            # Plot and log a sample prediction
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f" {test_name}, Error: {sample_error:.5f}")
            im1 = axs[0].imshow(sample_pred[0,-1,:,:,jj], cmap='viridis')
            axs[0].set_title(f' Prediction')
            fig.colorbar(im1, ax=axs[0])
            im2 = axs[1].imshow(sample_yy[0,-1,:,:,jj], cmap='viridis')
            axs[1].set_title(f' Ground Truth')
            fig.colorbar(im2, ax=axs[1])
            im3 = axs[2].imshow(np.abs(sample_yy[0,-1,:,:,jj] -sample_pred[0,0,:,:,jj]), cmap='viridis')
            axs[2].set_title(f' Diff')
            plt.tight_layout()
            fig.colorbar(im3, ax=axs[2])
            if args.use_writer:
                wandb.log({f"{test_name}_sample_plot_channel_{jj}": wandb.Image(fig)})
            plt.close(fig)

print(test_l2_steps)
print(test_l2_fulls)
for i in range(len(test_paths)):
    print('{}: {:.5f}, {:.5f}'.format(test_paths[i], test_l2_steps[i],test_l2_fulls[i]))
    if args.use_writer:
        wandb.log({f"test_loss_step_{test_paths[i]}": test_l2_steps[i], f"test_loss_full_{test_paths[i]}": test_l2_fulls[i]})

# Log the aggregated ns2d_pdb_M1 values
print('ns2d_pdb_M1: {:.5f}, {:.5f}'.format(step_pdb / num_samples, total_pdb / num_samples))
if args.use_writer:

    wandb.log({"ns2d_pdb_M1_step": step_pdb / num_samples, "ns2d_pdb_M1_full": total_pdb / num_samples})

# Log the total time and average time
print('Total time {} total steps {} Avg time {}'.format(time_test, total_steps, time_test / total_steps))
if args.use_writer:

    wandb.log({"total_time": time_test, "total_steps": total_steps, "avg_time_per_step": time_test / total_steps})
