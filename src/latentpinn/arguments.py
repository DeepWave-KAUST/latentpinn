from argparse import ArgumentParser


parser = ArgumentParser(description="Surface tomography example 1")
parser.add_argument(
    "--lateral_spacing",
    type=float,
    default=0.01,
    help="Lateral sampling.",
)
parser.add_argument(
    "--vertical_spacing",
    type=float,
    default=0.01,
    help="Vertical sampling.",
)
parser.add_argument(
    "--max_offset",
    type=float,
    default=5.0,
    help="Maximum offset.",
)
parser.add_argument(
    "--max_depth",
    type=float,
    default=1.0,
    help="Maximum depth.",
)
parser.add_argument(
    "--rec_spacing",
    type=int,
    default=10,
    help="Receiver sampling.",
)
parser.add_argument(
    "--sou_spacing",
    type=int,
    default=10,
    help="Source sampling.",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=250,
    help="Epochs length.",
)
parser.add_argument(
    "--num_neurons",
    type=int,
    default=20,
    help="Neurons width.",
)
parser.add_argument(
    "--num_layers",
    type=int,
    default=10,
    help="Layers depth.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-3,
    help="Learning rate.",
)
parser.add_argument(
    "--model_type",
    type=str,
    default="seam",
    help="Type of velocity model used.",
)
parser.add_argument(
    "--data_type",
    type=str,
    default="full",
    help="Type of data interpolation used.",
)
parser.add_argument(
    "--middle_shot",
    type=str,
    default="n",
    help="Whether the shots centered at the model (number of shot fixed) or spread accross.",
)
parser.add_argument(
    "--until_cmb",
    type=str,
    default="n",
    help="Whether the velocity reaches only down to the core-mantle boundary(CMB).",
)
parser.add_argument(
    "--earth_scale",
    type=str,
    default="n",
    help="Whether the experiment mimics the actual Earth's coordinate values.",
)
parser.add_argument(
    "--scale_factor",
    type=int,
    default=10,
    help="When the experiments are scaled this corresponds to the factor.",
)
parser.add_argument(
    "--reduce_after",
    type=int,
    default=15,
    help="When the learning rate should be reduced after stagnancy.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=123,
    help="Seed for reproducibility.",
)
parser.add_argument(
    "--initialization",
    type=str,
    default="varianceScaling",
    help="Seed for reproducibility.",
)
parser.add_argument(
    "--plotting_factor",
    type=int,
    default=1,
    help="The multiplication of the coordinate values (default is 1 km x 5 km, vertically and laterally).",
)
parser.add_argument(
    "--rescale_plot",
    type=str,
    default="n",
    help="Whether the plotting is scaled to mimic the Earth Crust experiment.",
)
parser.add_argument(
    "--depth_shift",
    type=str,
    default="n",
    help="Whether the plotting is shifted.",
)
parser.add_argument(
    "--tau_multiplier",
    type=float,
    default=3.0,
    help="Scaling factor for the last layer in the tau model.",
)
parser.add_argument(
    "--initial_velocity",
    type=float,
    default=4,
    help="Starting velocity.",
)
parser.add_argument(
    "--zid_source",
    type=int,
    default=5,
    help="Z index of the source location.",
)
parser.add_argument(
    "--zid_receiver",
    type=int,
    default=0,
    help="Z index of the receiver location.",
)
parser.add_argument(
    "--explode_reflector",
    type=str,
    default="n",
    help="Boolean whether to place the shots in the middle.",
)
parser.add_argument(
    "--field_synthetic",
    type=str,
    default="n",
    help="Boolean to mimic the field earthquake South California data.",
)
parser.add_argument(
    "--v_multiplier",
    type=float,
    default=3,
    help="Scaling factor for the last layer in the v model.",
)
parser.add_argument(
    "--activation",
    type=str,
    default="elu",
    help="Type of activation function.",
)
parser.add_argument(
    "--num_points",
    type=float,
    default=1.0,
    help="Fraction of the collocation points.",
)
parser.add_argument(
    "--irregular_grid",
    type=str,
    default="n",
    help="Boolean to select random non-regular grid training points.",
)
parser.add_argument(
    "--xid_well",
    type=int,
    default=5,
    help="Z index of the well location.",
)
parser.add_argument(
    "--last_vmultiplier",
    type=int,
    default=5,
    help="Integer scalar for the output of the velocity network.",
)
parser.add_argument(
    "--v_units",
    type=str,
    default="unitless",
    help="Set whether the velocity network predicts a scalar (unitless) qantity or the velocity field directly.",
)
parser.add_argument(
    "--well_depth",
    type=int,
    default=None,
    help="Depth index of the end point of the well.",
)
parser.add_argument(
    "--exp_function",
    type=str,
    default="n",
    help="Imposing smoothness through exponential function.",
)
parser.add_argument(
    "--exp_factor",
    type=float,
    default=1.0,
    help="Imposing smoothness through exponential function; its factor.",
)
parser.add_argument(
    "--exclude_topo",
    type=str,
    default="n",
    help="Whether the receiver stations locations are not a function of the topopgraphy.",
)
parser.add_argument(
    "--exclude_well",
    type=str,
    default="n",
    help="Boolean whether to include the well location for training.",
)
parser.add_argument(
    "--exclude_source",
    type=str,
    default="n",
    help="Boolean whether to include the source location for training.",
)
parser.add_argument(
    "--loss_function",
    type=str,
    default="mse",
    help="Type of metric for the regression loss.",
)
parser.add_argument(
    "--station_factor",
    type=float,
    default=1.0,
    help="Scale the real earthquake depth for semi-synhtetic experiments.",
)
parser.add_argument(
    "--event_factor",
    type=float,
    default=1.0,
    help="Scale the real earthquake depth for semi-synhtetic experiments.",
)
parser.add_argument(
    "--checker_size",
    type=float,
    default=5.0,
    help="Scale the real earthquake depth for semi-synhtetic experiments.",
)
parser.add_argument(
    "--tau_act",
    type=str,
    default="None",
    help="Last activation function for the tau model.",
)
parser.add_argument(
    "--empty_middle",
    type=str,
    default="n",
    help="Imposing no recording nor shot in the middle part of the model.",
)
parser.add_argument(
    "--factorization_type",
    type=str,
    default="multiplicative",
    help="Types of factorization used for the eikonal.",
)
parser.add_argument(
    "--causality_factor",
    type=float,
    default=1.0,
    help="Exponential coefficient from the original CausalPINNs paper.",
)
parser.add_argument(
    "--causality_weight",
    type=str,
    default="type_0",
    help="Exponential coefficient from the original CausalPINNs paper.",
)
parser.add_argument(
    "--residual_network",
    type=str,
    default="n",
    help="Whether a network with residual connections is used.",
)
parser.add_argument(
    "--velocity_loss",
    type=str,
    default="n",
    help="Whether the loss uses a velocity unit.",
)
parser.add_argument(
    "--regular_station",
    type=str,
    default="n",
    help="Whether the station for the field is regularly sampled.",
)
parser.add_argument(
    "--data_neurons",
    type=int,
    default=16,
    help="Neurons width.",
)
parser.add_argument(
    "--data_layers",
    type=int,
    default=8,
    help="Layers depth.",
)
parser.add_argument(
    "--append_shot",
    type=str,
    default="n",
    help="Whether we add virtual shot at the bottom right of the model.",
)
parser.add_argument(
    "--use_wandb",
    type=str,
    default="y",
    help="Whether we use weight and biases to keep track of the experiments.",
)
parser.add_argument(
    "--save_folder",
    type=str,
    default="./",
    help="Folder to keep track of the experiments when wandb is disabled.",
)
parser.add_argument(
    "--project_name",
    type=str,
    default="GFATT_PINNs-20-3d-lightning-inversion",
    help="The wandb project name when it is enabled.",
)
parser.add_argument(
    "--regularization_type",
    type=str,
    default="None",
    help="Types of regularization scheme (e.g., isotropic-TV, 1st-Tikhonov, 2nd-Tikhonov)",
)
parser.add_argument(
    "--regularization_weight",
    type=float,
    default=0.0,
    help="Regularization weighting coefficient",
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="adam",
    help="Type of optimization algorithm.",
)
parser.add_argument(
    "--mixed_precision",
    type=str,
    default="n",
    help="Whether the mixed precision flag is used.",
)
parser.add_argument(
    "--fast_loader",
    type=str,
    default="n",
    help="Whether a non-default PyTorch's data loader.",
)
parser.add_argument(
    "--sampling_rate",
    type=int,
    default=4,
    help="Sampling rate for the input velocity.",
)
parser.add_argument(
    "--initial_mean",
    type=float,
    default=0.5,
    help="Initialization weights.",
)
parser.add_argument(
    "--initial_bias",
    type=float,
    default=0.5,
    help="Initialization bias.",
)
parser.add_argument(
    "--initial_deviation",
    type=float,
    default=0.5,
    help="Initialization standard deviation.",
)
parser.add_argument(
    "--dual_optimizer",
    type=str,
    default="n",
    help="Whether dual optimizer is for each network used.",
)
parser.add_argument(
    "--tau_function",
    type=str,
    default="l2",
    help="Imposing smoothness through exponential function.",
)
parser.add_argument(
    "--v_function",
    type=str,
    default="l2",
    help="Type of norm used for the distance function in the velocity prediction.",
)
parser.add_argument(
    "--with_well",
    type=str,
    default="n",
    help="Boolean whether to include the well information for training.",
)
