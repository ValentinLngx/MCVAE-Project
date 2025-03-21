from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.nn as nn

from vaes import VAE, IWAE, AMCVAE, LMCVAE, VAE_with_flows, FMCVAE
from utils import make_dataloaders, get_activations, str2bool
from diffusion_decoder import FMCVAE_With_Diffusion

import torch
torch.set_float32_matmul_precision("high")

activation_dict = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU
}

if __name__ == '__main__':
    parser = ArgumentParser()
    #parser = pl.Trainer.add_argparse_args(parser)
    tb_logger = pl_loggers.TensorBoardLogger('lightning_logs/')

    parser.add_argument("--model", default="FMCVAE",
                        choices=["VAE", "IWAE", "AMCVAE", "LMCVAE", "VAE_with_flows", "FMCVAE", "FMCVAEWithDiffusion"])

    ## Dataset params
    parser.add_argument("--dataset", default='fashionmnist', choices=['mnist', 'fashionmnist', 'cifar', 'omniglot', 'celeba'])
    parser.add_argument("--binarize", type=str2bool, default=False)
    ## Training parameters
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--val_batch_size", default=128, type=int)
    parser.add_argument("--grad_skip_val", type=float, default=0.)
    parser.add_argument("--grad_clip_val", type=float, default=0.)

    ## Architecture
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--num_samples", default=1, type=int)
    parser.add_argument("--act_func", default="relu",choices=["relu", "leakyrelu", "tanh", "logsigmoid", "logsoftmax", "softplus", "gelu", "silu"])
    parser.add_argument("--net_type", choices=["fc", "conv"], type=str, default="conv")

    ## Specific parameters
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--n_leapfrogs", type=int, default=3)
    parser.add_argument("--step_size", type=float, default=0.001)

    parser.add_argument("--use_barker", type=str2bool, default=False)
    parser.add_argument("--use_score_matching", type=str2bool, default=False)  # for ULA
    parser.add_argument("--use_cloned_decoder", type=str2bool,
                        default=False)  # for AIS VAE (to make grad throught alphas easier)
    parser.add_argument("--learnable_transitions", type=str2bool,
                        default=False)  # for AIS VAE and ULA (if learn stepsize or not)
    parser.add_argument("--variance_sensitive_step", type=str2bool,
                        default=False)  # for AIS VAE and ULA (adapt stepsize based on dim's variance)
    parser.add_argument("--use_alpha_annealing", type=str2bool,
                        default=False)  # for AIS VAE, True if we want anneal sum_log_alphas during training
    parser.add_argument("--annealing_scheme", type=str,
                        default='linear')  # for AIS VAE and ULA VAE, strategy to do annealing
    parser.add_argument("--specific_likelihood", type=str,
                        default=None)  # specific likelihood

    parser.add_argument("--ula_skip_threshold", type=float,
                        default=0.0)  # Probability threshold, if below -- skip transition
    parser.add_argument("--acceptance_rate_target", type=float,
                        default=0.95)  # Target acceptance rate
    parser.add_argument("--sigma", type=float, default=1.)

    parser.add_argument("--num_flows", type=int, default=1)
    parser.add_argument("--Kprime", type=int, default=5,help="Number of AIS/SIS steps for FMCVAE")
    parser.add_argument("--ais_method", type=str, default="AIS",choices=["AIS", "SIS"],help="Weight update method for FMCVAE: 'AIS' accumulates weights, 'SIS' normalizes at each step.")
    parser.add_argument("--sampler_type", type=str, default="HMC",choices=["ULA", "MALA", "HMC"],
                    help="Markov transition sampler type for FMCVAE: ULA, MALA, or HMC.")
    parser.add_argument("--flow_type", type=str, default="RealNVP")

    parser.add_argument("--diffusion_steps", type=int, default=5, help="Toy diffusion steps for the refiner.")
    parser.add_argument("--diffusion_hidden_dim", type=int, default=128,help="Hidden dim in the toy diffusion refiner.")

    act_func = get_activations()

    args = parser.parse_args()
    print(args)


    kwargs = {'num_workers': 8, 'pin_memory': True}
    train_loader, val_loader = make_dataloaders(dataset=args.dataset,
                                                batch_size=args.batch_size,
                                                val_batch_size=args.val_batch_size,
                                                binarize=args.binarize,
                                                **kwargs)
    image_shape = train_loader.dataset.shape_size
    
    if args.model == "VAE":
        model = VAE(shape=image_shape, act_func=act_func[args.act_func],
                    num_samples=args.num_samples, hidden_dim=args.hidden_dim,
                    net_type=args.net_type, dataset=args.dataset, specific_likelihood=args.specific_likelihood,
                    sigma=args.sigma)
    elif args.model == "IWAE":
        model = IWAE(shape=image_shape, act_func=act_func[args.act_func], num_samples=args.num_samples,
                     hidden_dim=args.hidden_dim,
                     name=args.model, net_type=args.net_type, dataset=args.dataset,
                     specific_likelihood=args.specific_likelihood, sigma=args.sigma)
    elif args.model == "VAE_with_flows":
        model = VAE_with_flows(shape=image_shape, act_func=act_func[args.act_func], num_samples=args.num_samples,
                               hidden_dim=args.hidden_dim, name=args.model, flow_type="RealNVP",
                               num_flows=args.num_flows,
                               net_type=args.net_type, dataset=args.dataset,
                               specific_likelihood=args.specific_likelihood,
                               sigma=args.sigma)
    elif args.model == 'AMCVAE':
        model = AMCVAE(shape=image_shape, step_size=args.step_size, K=args.K, use_barker=args.use_barker,
                       num_samples=args.num_samples, acceptance_rate_target=args.acceptance_rate_target,
                       dataset=args.dataset, net_type=args.net_type, act_func=act_func[args.act_func],
                       hidden_dim=args.hidden_dim, name=args.model, grad_skip_val=args.grad_skip_val,
                       grad_clip_val=args.grad_clip_val,
                       use_cloned_decoder=args.use_cloned_decoder, learnable_transitions=args.learnable_transitions,
                       variance_sensitive_step=args.variance_sensitive_step,
                       use_alpha_annealing=args.use_alpha_annealing, annealing_scheme=args.annealing_scheme,
                       specific_likelihood=args.specific_likelihood, sigma=args.sigma)
    elif args.model == 'LMCVAE':
        model = LMCVAE(shape=image_shape, step_size=args.step_size, K=args.K,
                       num_samples=args.num_samples, acceptance_rate_target=args.acceptance_rate_target,
                       dataset=args.dataset, net_type=args.net_type, act_func=act_func[args.act_func],
                       hidden_dim=args.hidden_dim, name=args.model, grad_skip_val=args.grad_skip_val,
                       grad_clip_val=args.grad_clip_val, use_score_matching=args.use_score_matching,
                       use_cloned_decoder=args.use_cloned_decoder, learnable_transitions=args.learnable_transitions,
                       variance_sensitive_step=args.variance_sensitive_step,
                       ula_skip_threshold=args.ula_skip_threshold, annealing_scheme=args.annealing_scheme,
                       specific_likelihood=args.specific_likelihood, sigma=args.sigma)

    elif args.model == "FMCVAE":
        model = FMCVAE(
            shape=image_shape, act_func=act_func[args.act_func],num_samples=args.num_samples,
            hidden_dim=args.hidden_dim, name="FMCVAE",flow_type=args.flow_type, num_flows=args.num_flows,
            net_type="conv",  dataset=args.dataset,specific_likelihood=None,sigma=1.0,
            Kprime=args.Kprime,sampler_type=args.sampler_type,sampler_step_size=args.step_size,ais_method=args.ais_method)
    
    elif args.model == "FMCVAEWithDiffusion":
        model = FMCVAE_With_Diffusion(shape=image_shape, act_func=act_func[args.act_func],num_samples=args.num_samples,
            hidden_dim=args.hidden_dim, name="FMCVAEWithDiffusion", flow_type=args.flow_type, num_flows=args.num_flows,
            net_type="conv", dataset=args.dataset,specific_likelihood=None, sigma=1.0, Kprime=args.Kprime, sampler_type=args.sampler_type,
            sampler_step_size=args.step_size, ais_method=args.ais_method,
            diffusion_steps=args.diffusion_steps,  diffusion_hidden_dim=args.diffusion_hidden_dim)

    else:
        raise ValueError

    args.gradient_clip_val = args.grad_clip_val
    automatic_optimization = (args.grad_skip_val == 0.) and (args.gradient_clip_val == 0.)

    trainer_kwargs = {
        "logger": tb_logger,
        "fast_dev_run": False,
        "accelerator": "gpu",
        "devices": 1,
        "max_epochs": 50  # number of epochs
        # "terminate_on_nan": automatic_optimization,  # Remove or comment out this line
    }
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Set model to evaluation mode
    model.eval()

    # Number of samples to generate
    n_samples = 16  # adjust as needed

    # Sample from the latent space (assume model.hidden_dim is the latent dimension)
    z = torch.randn(n_samples, model.hidden_dim).to(next(model.parameters()).device)

    # Generate images by passing z through the model (or model.decoder, if you have a dedicated decoder)
    with torch.no_grad():
        #generated_images = model(z)
        generated_images = model.decode(z)
        # If the model outputs logits, apply a sigmoid activation to get values in [0,1]
        generated_images = torch.sigmoid(generated_images)

    # Create a grid of images
    grid = vutils.make_grid(generated_images, nrow=4, normalize=True, scale_each=True)

    # Display the grid using matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Generated MNIST Samples")
    plt.axis("off")
    plt.savefig("generated_samples.png")
    plt.show()