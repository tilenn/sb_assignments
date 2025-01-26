from modules.irisRecognition import irisRecognition
from modules.utils import get_cfg
import argparse
import glob
from PIL import Image
import os
import cv2
import numpy as np
import torch

def main(cfg):

    PRELOAD_TEMPLATES = False

    irisRec = irisRecognition(cfg)

    if not os.path.exists('./dataProcessed/'):
        os.mkdir('./dataProcessed/')
    
    if not os.path.exists('./templates/'):
        os.mkdir('./templates/')

    # Get the list of images to process
    print("Processing images in ./data/ ...")
    filename_list = [os.path.basename(fn) for fn in glob.glob("./data/*.jpg")]
    image_list = []

    # Segmentation, normalization and encoding
    polar_mask_list = []
    code_list = []

    if not PRELOAD_TEMPLATES:
        for filename in glob.glob("./data/*.jpg"):
            im = Image.fromarray(np.array(Image.open(filename).convert("RGB"))[:, :, 0], "L")
            image_list.append(im)
            # filename_list.append(os.path.basename(filename))

    else:   
        # load templates
        for fn in filename_list:
            template = np.load("./templates/" + os.path.splitext(fn)[0] + "_tmpl.npz")
            # get the data
            template = template['arr_0']
            # convert to torch tensor
            template = torch.from_numpy(template)
            code_list.append(template)

    if not PRELOAD_TEMPLATES:
        for im,fn in zip(image_list,filename_list):
            
            print(fn)

            # convert to ISO-compliant aspect ratio (4:3) and resize to ISO-compliant resolution: 640x480
            im = irisRec.fix_image(im)

            # segmentation mask and circular approximation:
            mask, pupil_xyr, iris_xyr = irisRec.segment_and_circApprox(im)
            im_mask = Image.fromarray(np.where(mask > 0.5, 255, 0).astype(np.uint8), 'L')

            # cartesian to polar transformation:
            im_polar, mask_polar = irisRec.cartToPol_torch(im, mask, pupil_xyr, iris_xyr)
            polar_mask_list.append(mask_polar)

            # im_polar has shape of (64, 512)
            # same goes for mask_polar

            # human-driven BSIF encoding:
            code = irisRec.extractIBBCode(im_polar)  #[, mask]) < masks are up to you where to use them

            code_list.append(code)

            # DEBUG: save selected processing results
            im_mask.save("./dataProcessed/" + os.path.splitext(fn)[0] + "_seg_mask.png")
            imVis = irisRec.segmentVis(im,mask,pupil_xyr,iris_xyr)
            path = "./dataProcessed/" + os.path.splitext(fn)[0]
            cv2.imwrite(path + "_seg_vis.png",imVis)
            cv2.imwrite(path + "_im_polar.png",im_polar)
            cv2.imwrite(path + "_mask_polar.png",mask_polar)
            np.savez_compressed("./templates/" + os.path.splitext(fn)[0] + "_tmpl.npz",code)
            # for i in range(irisRec.num_filters):
                # cv2.imwrite(("%s_code_filter%d.png" % (path,i)),255*code[i,:,:])
    
    # Matching (all-vs-all) and computing metrics
    scores = []
    labels = []
    current_match = 1 # for printing progress
    num_matches = len(code_list) * (len(code_list)-1) / 2
    for code1,fn1,i in zip(code_list,filename_list,range(len(code_list))):
        for code2,fn2,j in zip(code_list,filename_list,range(len(code_list))):
            if i < j:
                score = irisRec.matchIBBCodes(code1, code2) #[, mask1, mask2]) < masks are up to you where to use them
                
                # Save score and label (genuine=1 or impostor=0)
                scores.append(score)
                same_iris = fn1.split('_')[1] == fn2.split('_')[1]
                labels.append(1 if same_iris else 0)

                print("{} <-> {} : {:.3f}, label: {} ({}/{})".format(fn1,fn2,score,1 if same_iris else 0, current_match, num_matches))
                current_match += 1
    
    # Convert to numpy arrays for easier computation
    scores = np.array(scores)
    labels = np.array(labels)

    # Calculate EER
    FARs = []
    FRRs = []
    thresholds = np.linspace(0, 1, 1000)
    for threshold in thresholds:
        predictions = (scores <= threshold).astype(int)
        TP = np.sum((predictions == 1) & (labels == 1))
        TN = np.sum((predictions == 0) & (labels == 0))
        FP = np.sum((predictions == 1) & (labels == 0))
        FN = np.sum((predictions == 0) & (labels == 1))
        
        far = FP / (FP + TN) if (FP + TN) > 0 else 0
        frr = FN / (FN + TP) if (FN + TP) > 0 else 0
        
        FARs.append(far)
        FRRs.append(frr)
    
    # Find the threshold where FAR almost equals FRR
    FARs = np.array(FARs)
    FRRs = np.array(FRRs)
    eer_idx = np.argmin(np.absolute(FARs - FRRs))
    eer_threshold = thresholds[eer_idx]
    eer_value = (FARs[eer_idx] + FRRs[eer_idx]) / 2
    
    # Calculate the classification accuracy for the `eer_threshold`
    predictions_eer = (scores <= eer_threshold).astype(int)
    TP_eer = np.sum((predictions_eer == 1) & (labels == 1))
    TN_eer = np.sum((predictions_eer == 0) & (labels == 0))
    accuracy_eer = (TP_eer + TN_eer) / len(labels)

    print("Results:")
    print("Threshold: {:.6f}".format(eer_threshold))
    print("EER: {:.6f}".format(eer_value))
    print("FAR: {:.6f}".format(FARs[eer_idx]))
    print("FRR: {:.6f}".format(FRRs[eer_idx]))
    print("Accuracy: {:.6f}".format(accuracy_eer))
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",
                        type=str,
                        default="cfg.yaml",
                        help="path of the configuration file")
    args = parser.parse_args()
    main(get_cfg(args.cfg_path))