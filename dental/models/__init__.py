from .base_model import *

def load_Generator(args):
    if args.model=="base":
        model=BaseGenerator(n_res=args.n_res)
    
    if args.resize:
        model=Resizing_Generator(n_res=args.n_res) # current tackle only 256 size

    return model

def load_Discriminator(args):
    if args.model=="base":
        model=BaseDiscriminator()

    return model