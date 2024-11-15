import os
import json
import math
import timm
import torch
import argparse
import datetime
import numpy as np
import torch.nn as nn
import loralib as lora

from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from dataloader import pb2_dataset

from pb2_model import Transformer_lora
from tokenizer import BPETokenizer
from evaluate import eval
import matplotlib.pyplot as plt
import torch.nn.functional as F
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device ="cpu"
    print("device:", device)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config = {
        "data_dir": args.data_dir,
        "val_dir": args.val_dir,
        # "pred_file": args.pred_file,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "decoder_weights": args.decoder_weights,
        "save_dir": args.save_dir,
    }
    print("config:", config)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_dir_with_time = os.path.join(args.save_dir, current_time)
    if not os.path.exists(save_dir_with_time):
        os.makedirs(save_dir_with_time)
    pred_file = os.path.join(save_dir_with_time, "output.json")
    encoder_model_name = "vit_gigantic_patch14_clip_224"
    # encoder_model_name = "vit_large_patch14_clip_224"
    # encoder_model_name = "vit_huge_patch14_clip_224"
    # encoder_model_name = "vit_large_patch14_clip_336"
    encoder = timm.create_model(encoder_model_name, pretrained=True)
    model = Transformer_lora(encoder, args.decoder_weights).to(device)
    # model_path = "/home/phank1026/dlcv/dlcv-fall-2024-hw3-hank09901/p2_save_dir/20241114-095105/CIDER_0.9611_CLIP_0.7142_epoch_9_loss_2.5062.ckpt"
    # model_path ="/home/phank1026/dlcv/dlcv-fall-2024-hw3-hank09901/p2_save_dir/CIDER0.9068_CLIP_0.71/CIDER_0.8947_CLIP_0.7030_epoch_8_loss_2.7536.ckpt"
    # model_path ='/home/phank1026/dlcv/dlcv-fall-2024-hw3-hank09901/p2_save_dir/ciser0.7213CLIP0.6840/CIDER_0.7213_CLIP_0.6840_epoch_6_loss_3.0345.ckpt' # use this to get best results without lora_alpha
    # model_path = "/home/phank1026/dlcv/dlcv-fall-2024-hw3-hank09901/p2_save_dir/cider0.8185_CLIP0.6955/CIDER_0.7747_CLIP_0.6895_epoch_5_loss_2.9129.ckpt"
    # model.load_state_dict(torch.load(model_path, device), strict=False)
    lora.mark_only_lora_as_trainable(model)

    for name, param in model.decoder.named_parameters():
        # print("name:", name)
        # print("param:", param)
        # if 'cross_attn' in name or 'ln_2' in name or 'linear' in name:
        if "linear" in name:
            # save train layer name in txt
            with open(os.path.join(save_dir_with_time, "train_layer.txt"), "w") as f:
                f.write(name + "\n")
            param.requires_grad = True

    trainable_weights = [name for name, param in model.named_parameters() if param.requires_grad == True]
    with open(os.path.join(save_dir_with_time, "train_layer.txt"), "a") as f:
        f.write("trainable_weights: " + str(trainable_weights ) + "\n")
        
    save_weights = {k: v for k,v in model.state_dict().items() if k in trainable_weights}
    print('Total params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    encoder_file = './encoder.json'
    vocab_file = './vocab.bpe'
    tokenizer = BPETokenizer(encoder_file, vocab_file)

    json_dir = "./hw3_data/p2_data"
    data_dir = args.data_dir
    val_dir = args.val_dir

    train_dataset = pb2_dataset(data_dir=data_dir, json_dir=json_dir, mode="train")
    val_dataset = pb2_dataset(data_dir=val_dir, json_dir=json_dir, mode="val", batch_size=1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-5)

    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader)*args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    writer = SummaryWriter()
    train_loss_list = []
    val_loss_list = []
    # best_val_loss = float('inf')
    best_epoch = 0
    lr_list = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, (image_name, images, captions_in, captions_gt) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            captions_in = captions_in.to(device)
            captions_gt = captions_gt.to(device)
            # print("image_name:", image_name)
            # print("captions_in:", captions_in)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images, captions_in)
                loss = criterion(outputs.view(-1, outputs.size(-1)), captions_gt.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            if not math.isnan(loss.item()):
                total_loss += loss.item()
            else:
                print(f"nan loss at epoch {epoch}, iter {i}")
                

        train_loss_list.append(total_loss / len(train_loader)) 
        lr_list.append(optimizer.param_groups[0]['lr'])
        # plot train_loss_list with label x = epoch, y = loss 
        print("[ Train | {}/{} ] loss = {:.4f}".format(epoch + 1, args.epochs, total_loss / len(train_loader)))
        plt.plot(range(len(train_loss_list)), train_loss_list)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train Loss")
        plt.savefig(os.path.join(save_dir_with_time, "train_loss.png"))
        plt.close()


        
        # plot lr_list with label x = epoch, y = lr
        plt.plot(range(len(lr_list)), lr_list)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate")
        plt.savefig(os.path.join(save_dir_with_time, "lr.png"))
        plt.close()
        writer.add_scalar("Loss/train", total_loss / len(train_loader), epoch)
        # ref :print(f"[ Train | {epoch + 1:02d}/{n_epochs:02d} ] loss = {train_loss_avg:.4f}")
        print(f"[ Train | {epoch + 1:02d}/{args.epochs:02d} ] loss = {total_loss / len(train_loader):.4f}")
        
        json_dict = {}
        if epoch>=2:
            model.eval()
            with torch.no_grad():
                for (image_name, images)in tqdm(val_loader):
                    batch_cnt = images.size(0)
                    img = images.to(device)

                    
                    yhat = model.autoreg_infer_beam(img, beam=4, step=20)
                

                    for i in range(batch_cnt):
                        name =  image_name[i]

                        yhat_sub = yhat[i].detach().cpu().tolist()[1:]
                        try:
                            y_hat_endidx = yhat_sub.index(50256)
                        except:
                            y_hat_endidx = len(yhat_sub)
                        yhat_sub = yhat_sub[:y_hat_endidx]

                        predict = tokenizer.decode(yhat_sub)
                        
                        # filecnt += 1
                        json_dict[image_name[i]] = predict
        # with torch.no_grad():
            #:
                # for (image_name, images) in tqdm(val_loader):
                    
                #     images = images.to(device)
                #     print("image_name:", image_name[0])
                    
                    # pred_token = [50256]
                    # while len(pred_token) <= 1 or pred_token[-1] != 50256:
                    #     if len(pred_token) == 100:
                    #         break
                    #     features = model.encoder.forward_features(images)
                    #     logits = model.decoder(features, torch.tensor(pred_token).unsqueeze(0).to(device))
                    #     # logits = model(images, torch.tensor(pred_token).unsqueeze(0).to(device))
                    #     prob = F.softmax(logits[:, -1, :], dim=-1)
                    #     token = torch.argmax(prob, dim=-1).view(-1).tolist()[-1]
                    #     pred_token.append(int(token))
                    
                    # pred_caption = tokenizer.decode(pred_token)
                    # pred_word = pred_caption.replace('<|endoftext|>', '')
                    # prediction[str(img_name)] = pred_caption
                # for (image_name, images)  in tqdm(val_loader):
                #     images = images.to(device)
                #     output_ids = model.beam_search(images, beams=3, max_length=100)
                    
                #     print("image_name:", image_name[0])
                    

                #     pred_word = tokenizer.decode(output_ids)
                #     print("pred_word:", pred_word)
                # # #     print("image_name:", image_name)
                # # #     # detuple image_name
                    
                #     json_dict[image_name[0]] = pred_word
                print("json_dict:", json_dict)
                with open(pred_file, "w") as f:
                    json.dump(json_dict, f, indent=4)
                cider_score, clip_score = eval(pred_file, val_dir, os.path.join(json_dir, "val.json"))
                trainable_weights = [name for name, param in model.named_parameters() if param.requires_grad == True]
                save_weights = {k: v for k,v in model.state_dict().items() if k in trainable_weights}
                torch.save(save_weights, os.path.join(save_dir_with_time, f'CIDER_{cider_score:.4f}_CLIP_{clip_score:.4f}_epoch_{epoch}_loss_{train_loss_list[-1]:.4f}.ckpt'))
                    #    f'CIDER_{cider_score:.4f}_CLIP_{clip_score:.4f}_epoch_{epoch}_loss_{train_loss_list[-1]:.4f}.ckpt')
            # torch.save(save_weights, os.path.join(save_dir_with_time, f'epoch_{epoch}_loss_{train_loss_list[-1]:.4f}.ckpt'))
                
            # except Exception as e:
                # print("Error:", e)
                # trainable_weights = [name for name, param in model.named_parameters() if param.requires_grad == True]
                # save_weights = {k: v for k,v in model.state_dict().items() if k in trainable_weights}
                # # torch.save(save_weights, f'CIDER_{cider_score:.4f}_CLIP_{clip_score:.4f}_epoch_{epoch}_loss_{train_loss_list[-1]:.4f}.ckpt')
                # torch.save(save_weights, os.path.join(save_dir_with_time, f'epoch_{epoch}_loss_{train_loss_list[-1]:.4f}.ckpt'))
                # continue
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./hw3_data/p2_data/images/train", help="path to images root directory")
    parser.add_argument("--val_dir", type=str, default="./hw3_data/p2_data/images/val", help="path to images root directory")
    # parser.add_argument("--pred_file", type=str, default="./save_dir/output.json", help="path to prediction file")
    # parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    # parser.add_argument("--num_workers", type=int, default=8, help="number of workers")
    parser.add_argument("--num_workers", type=int, default=2, help="number of workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--seed", type=int, default=1026, help="seed")
    parser.add_argument("--decoder_weights", type=str, default="./hw3_data/p2_data/decoder_model.bin", help="path to decoder weights")
    parser.add_argument("--save_dir", type=str, default="./p2_save_dir", help="path to save directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    main(args)
    # Load