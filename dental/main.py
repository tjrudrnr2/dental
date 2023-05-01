import models
import argparse
import torch
from models import load_Generator,load_Discriminator
from opt import get_opts
from trainer import Trainer
import torchvision.utils as tvutils
from torchvision import transforms


from torch.utils.data import DataLoader
from datasets.dataloader import get_train_loader,get_test_loader,get_eval_loader
import numpy as np
import os
import wandb

def define_models(args):
    G=load_Generator(args)
    F=load_Generator(args)
    D_y=load_Discriminator(args)
    D_x=load_Discriminator(args)
    return G,F,D_y,D_x

def define_generator(args):
    G=load_Generator(args)
    F=load_Generator(args)
    return G,F

def load_pretrained_generators(G, F, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    G.load_state_dict(checkpoint['G_state_dict'])
    F.load_state_dict(checkpoint['F_state_dict'])


def main():
    torch.cuda.empty_cache()
    args=get_opts()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.test:
        assert args.model_path, 'model_path must be provided for testing'
        ## TODO : 여기는 원하시는대로 test코드 짜서 쓰셔용~!

        print("Testing...")
        G,F=define_generator(args)
        G=G.to(device)
        F=F.to(device)
        G.eval()
        F.eval()

        load_pretrained_generators(G,F,args.model_path)
        
        test_loader = get_test_loader(root=args.root_dir, batch_size=args.batch_size, shuffle=False,resize=args.resize,gray=args.gray)
        image_batch=next(iter(test_loader)).to(device)
        new_images=G(image_batch).detach().cpu()

        tvutils.save_image(image_batch, 'test_images.jpg', nrow=3, padding=2, normalize=True, value_range=(-1, 1))
        tvutils.save_image(new_images, 'generated_images.jpg', nrow=3, padding=2, normalize=True, value_range=(-1, 1))
        # for generate sample:
        eval_loader=get_eval_loader(root=args.root_dir, batch_size=args.batch_size, shuffle=False,gray=args.gray)
        '''
        for image_batch in  eval_loader:
            image_batch=image_batch.to(device)
            generated_images=G(image_batch).to(device)
            for generated_image in generated_images:
                
           '''   

    else:
        print("Training...")
        if args.save:
            wandb.init(project=args.project_name,reinit=True)
            wandb.config.update(args)
            if args.run_name:
                wandb.run.name=args.run_name
                wandb.run.save()
        print("model_loading...")
        G,F,D_y,D_x=define_models(args)
        
        loader,target_loader=get_train_loader(args.root_dir,args.target_dir,batchsize=args.batch_size,resize=args.resize)
        print("loader len : ", len(loader))
        print("target_loader len : ", len(target_loader))
        trainer=Trainer(G,F,D_y,D_x,loader,target_loader,device,args)
        if args.model_path:
            trainer.load_checkpoint(args.model_path)
        save_path = os.path.join(args.generated_image_save_path, f'{args.run_name}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print('Start Training...')
        loss_D_x_hist, loss_D_y_hist, loss_G_GAN_hist, loss_F_GAN_hist, \
        loss_cycle_hist, loss_identity_hist=trainer.train(num_epochs=args.num_epochs,initialization_epochs=args.initialization_epochs,save_path=save_path)
        
        if args.save:
            wandb.log({"loss_D_x_hist": wandb.Histogram(loss_D_x_hist),
                    "loss_D_y_hist" : wandb.Histogram(loss_D_y_hist),
                    "loss_G_GAN_hist" : wandb.Histogram(loss_G_GAN_hist),
                    "loss_F_GAN_hist" : wandb.Histogram(loss_F_GAN_hist),
                    "loss_cycle_hist" : wandb.Histogram(loss_cycle_hist),
                    "loss_identity_hist" : wandb.Histogram(loss_identity_hist)})
        # 시험용으로 해봄
        test_images = get_test_loader(root=args.root_dir, batch_size=args.batch_size, shuffle=False,resize=args.resize)
        image_batch= next(iter(test_images))
        image_batch = image_batch.to(device)
        new_images = G(image_batch).detach().cpu()
        
        test_images = get_test_loader(root=args.target_dir, batch_size=args.batch_size, shuffle=False,resize=args.resize)
        image_batch= next(iter(test_images))
        image_batch = image_batch.to(device)
        new_images_2 = F(image_batch).detach().cpu()

        tvutils.save_image(image_batch, 'test_images.jpg', nrow=3, padding=2, normalize=True, value_range=(-1, 1))
        tvutils.save_image(new_images, f'generated_X_{args.loss}.jpg', nrow=3, padding=2, normalize=True, value_range=(-1, 1))   
        tvutils.save_image(new_images_2, f'generated_Y_{args.loss}.jpg', nrow=3, padding=2, normalize=True, value_range=(-1, 1))
        
    # define_model

if __name__=="__main__":
    main()
