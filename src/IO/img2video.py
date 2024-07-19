import sys
import cv2
import glob
from natsort import natsorted
import os
from gooey import GooeyParser


parser = GooeyParser(description='Regress texture from pose')

parser.add_argument(
        '--folder_path',
        default= None,
        type=str,
        help='Where pose_path')

parser.add_argument(
        '--fps',
        default= 18,
        type=float,
        help='Where pose_path')

args = parser.parse_args()

folder_path = args.folder_path

#folder_path = r"D:\Project\Human\Avatar-In-The-Shell\Result\result\20211115_SCANimate_data_collection\Processing_lists\8_Muscle_range_of_motion\test\release_muscle_range_of_motion_test_Taunt\test_Taunt\seqs\rendering"
img_folder_path = natsorted(glob.glob(os.path.join(folder_path ,  r"*.png")))
save_path = os.path.join(folder_path , r"rendering.mp4")
print(img_folder_path)
print(save_path)


#img_folder_path = natsorted(glob.glob(r"D:\Project\Human\Avatar-In-The-Shell\Result\result\rendering\posed\*"))
#img_folder_path = natsorted(glob.glob(r"D:\Project\Human\Avatar-In-The-Shell\Result\result\20211115_SCANimate_data_collection\Processing_lists\8_Muscle_range_of_motion\test\release_muscle_range_of_motion_test_Taunt\test_Taunt\seqs\rendering\*.png"))
#img_folder_path = natsorted(glob.glob(r"D:\Project\Human\Avatar-In-The-Shell\Result\result\rendering\posed\rendering_*.png"))
#img_folder_path = natsorted(glob.glob(r"D:\Project\Human\Avatar-In-The-Shell\Result\result\rendering\posed_color\rendering*.png"))
#img_folder_path = natsorted(glob.glob(r"D:\Data\Human\HUAWEI\Cape_fitted_00159\data\texture_rgb\displacement_texture_*"))
#fps = 1.0
#fps = 5.0
#fps = 10.0
fps = args.fps
#save_path = r"D:\Project\Human\Avatar-In-The-Shell\Result\result\rendering\rendering.mp4"
#save_path = r"D:\Project\Human\Avatar-In-The-Shell\Result\result\rendering\rendering.avi"
#save_path = r"D:\Data\Human\HUAWEI\Iwamoto\data\texture_rgb_globalbasis\textures.mp4"
#dataname = "HuaweiData "
#dataname = "MIT I_crane"
#dataname = "CAPE       "
dataname = ""

# encoder(for mp4)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 's')
#fourcc = cv2.VideoWriter_fourcc('m', 'j', 'p', 'g')
#fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '5')
#fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
#fourcc = cv2.VideoWriter_fourcc('X', '2', '6', '4')
# output file name, encoder, fps, size(fit to image size)
#video = cv2.VideoWriter(save_path,fourcc, fps, (1920, 1080))
#video = cv2.VideoWriter(save_path,fourcc, fps, (1920, 1920))
#video = cv2.VideoWriter(save_path,fourcc, fps, (960, 540))
#video = cv2.VideoWriter(save_path,fourcc, fps, (256,256))
video = cv2.VideoWriter(save_path,fourcc, fps, (768, 576))


if not video.isOpened():
    print("can't be opened")
    sys.exit()


for i,img_path in enumerate(img_folder_path):
    #print(img_path)
    img = cv2.imread(img_path)
    #print(img.shape)
    #frame_id = i + 2
    frame_id = int(os.path.basename(img_path).split(".")[0].split("_")[-1])
    #if (frame_id<70):
    #    continue
    #if (frame_id%10!=5):
    #   continue
    print(frame_id , ":" ,img_path)

    #write frame_id
    cv2.putText(img,
            #text= "dataset : " + dataname + "  " + str(fps) + "fps  id :" + str(frame_id),
            text= "id :" + str(frame_id),
            #text= "dataset : " + "avgtexture_alpha3_rgb" + "  " + str(fps) + "fps  id :" + str(frame_id),
            org=(30,30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(255, 255, 255),
            thickness=2,
            #thickness=1,
            lineType=cv2.LINE_4)


    # can't read image, escape
    if img is None:
        print("can't read")
        break

    # add
    video.write(img)

video.release()
print('written')