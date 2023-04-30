import argparse
import glob
import os
import cv2

def config_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataroot',type=str, default="../data/R",
                        help="data_root")
    parser.add_argument('--cut_off',default=True,action='store_false')
    parser.add_argument('--cut_off_size',default="R")
    parser.add_argument('--save_dir',type=str, default="./data/processed" )

    args=parser.parse_args()
    return args

def cut_off(im):
    
    return 

if __name__=="__main__":
    args=config_parser()
    img_list=glob.glob(os.path.join(args.dataroot,"*.jpg"))
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    
    
    for path in img_list:

        im=cv2.imread(path)
        # R cut_off (150,2790) (85,1325)
        # V cut_off(230,2770) (105,1345)
        if args.cut_off:
            if args.cut_off_size=="R":
                im=im[95:1335,170:2770]
            else:
                im=im[100:1340,200:2800]

        cv2.imwrite(os.path.join(args.save_dir,path[-12:]),im)

        
            
        