# This script aligns GST images with HMI images
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import gst_alignment_tools as gstt
import sys
import warnings
warnings.filterwarnings('ignore')
#######################################################################################################
########################## Parameters adjustment begin ########################################
############################################################################

#------------------Data preparation-----------------------
# resize the gst_image to the HMI images' resolution when '_tar_resized' is True(recommanded)
_tar_resized=True
#HMI fits for most of the time.
ref_fits = '/Volumes/Data/from_qq/hmi.Ic_45s.20200611_175230_TAI.2.continuum.fits'
#Please provide a GST fits file without any further process after it comes out of GST's pipeline.
#gst_fits = ''
gst_fits = '/Volumes/Data/from_qq/bbso_tio_pcosr_20200611_175248.fts'
#tmp_dir is your local foder which contains all the processed out put.
tmp_dir = '/Volumes/Data/from_qq/tmp'
# Make GST's fits header has similar look as the product of HMI
# rescale the images to enhance the featrue
# ------------------------------image enhancement and ROI selection-------------------------------------------------------
br1 = [190, 230]
br2 = [170, 230]
gamma1=0.6
gamma2=0.6
#-----mask out the edge. Use the central region of the image(set _center_mask1/2 to (0.0 ~ 1.0)) e.g. center_mask1/2 = 1.0( no mask), center_mask1/2 = 0.5(use the central area with half the height and width)
#---or you can define your own mask--------------
_center_mask1 = 0.5
_center_mask2 = 0.4
# --------------------Feature calculation-------------------------
MAX_MATCHES = 70
SCALE_FACTOR = 1.10
#-----------------Featrue Matching-------------------
mratio = 0.77

#######################################################################################################
########################## Parameters adjustment end ########################################
############################################################################
processed_gst_fits = gstt.fits_header_modifier_gst(filename=gst_fits, tmp_dir=tmp_dir)
#Prepare the numpy array for alignment
#resize the gst_image to the HMI images' resolution when 'tar_resized' is True(recommanded)
cur_data, ref_data, cur_map, ref_map = gstt.compared_arraies(fitsfile=processed_gst_fits,ref_fitsfile=ref_fits,tar_resized=_tar_resized)
#----------Image enhancement and ROI selection-------------------------------------------
im1Gray = np.array(cur_data * 255 / np.nanmax(cur_data), dtype=np.uint8)
im2Gray = np.array(ref_data * 255 / np.nanmax(ref_data), dtype=np.uint8)
histogram1 = cv2.calcHist([im1Gray], [0], None, [256], [0, 256])
histogram2 = cv2.calcHist([im2Gray], [0], None, [256], [0, 256])
fig_his, axs_his = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
axs_his[0].set_ylim([0, 2500])
axs_his[1].set_ylim([0, 2500])
fig_data, axs_data = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
axs_data[0].imshow(im1Gray, origin='lower')
axs_data[1].imshow(im2Gray, origin='lower')
plt.show()
#----------------------if you want to stop and enhance the images-----------------------------------
yes = {'yes', 'y', 'ye', ''}
no = {'no', 'n'}

print('Do you want to stop and enhance the images? ')
choice = input().lower()
if choice in yes:
    sys.exit()
elif choice in no:
    pass
else:
    sys.stdout.write("Please respond with 'yes' or 'no'")
#------------------------------image enhancement and ROI selection-------------------------------------------------------
im1Gray = gstt.contrast_adjustment(im1Gray, br1, gamma=gamma1) #rescale the images to enhance the featrue. Otherwise, comment it out.
im2Gray = gstt.contrast_adjustment(im2Gray, br2, gamma=gamma2) #rescale the images to enhance the featrue. Otherwise, comment it out.
#-----mask out the edge. Use the central region of the image(set _center_mask1/2 to (0.0 ~ 1.0)) e.g. center_mask1/2 = 1.0( no mask), center_mask1/2 = 0.5(use the central area with half the height and width)
#---or you can define your own mask--------------
mask1, mask2 = gstt.define_mask(center_mask1=_center_mask1, center_mask2=_center_mask2, im1 = im1Gray, im2 = im2Gray)
masked_im1 = cv2.bitwise_and(im1Gray, im1Gray, mask=mask1)
masked_im2 = cv2.bitwise_and(im2Gray, im2Gray, mask=mask2)
fig_edata, axs_edata = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
axs_edata[0].imshow(masked_im1, origin='lower')
axs_edata[1].imshow(masked_im2, origin='lower')
plt.show()

#--------------------Image alignment-------------------------
MAX_MATCHES = 70
SCALE_FACTOR = 1.10
orb = cv2.ORB_create(MAX_MATCHES, scaleFactor=SCALE_FACTOR)
#Calculate features in each image
keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, mask1, None)
keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, mask2, None)
#Draw the features on each image
kp1_image = cv2.drawKeypoints(im1Gray, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kp2_image = cv2.drawKeypoints(im2Gray, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
fig_feaure, axs_feature = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
axs_feature[0].imshow(kp1_image, origin='lower')
axs_feature[1].imshow(kp2_image, origin='lower')
plt.show()
cv2.imwrite(os.path.join(tmp_dir,"kp1.jpg"), kp1_image)
cv2.imwrite(os.path.join(tmp_dir,"kp2.jpg"), kp2_image)
#Match the features
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
print('there are {} matches in total'.format(len(matches)))
good = []
good_without_list = []
for m, n in matches:
    if m.distance < mratio * n.distance:
        good.append([m])
        good_without_list.append(m)
img_match = cv2.drawMatchesKnn(im1Gray, keypoints1, im2Gray, keypoints2, good, None, flags=2)
cv2.imwrite(os.path.join(tmp_dir, "match.jpg"), img_match)
fig_match, axs_match = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
axs_match.imshow(img_match, origin='lower')
plt.show()
print('Does matching look OK?')
choice = input().lower()
if choice in yes:
    pass
elif choice in no:
    sys.exit()
else:
    sys.stdout.write("Please respond with 'yes' or 'no'")
#----------------------------calculate the transfer Matrix---------------------------
# Extract location of good matches
points1 = np.zeros((len(good_without_list), 2), dtype=np.float32)
points2 = np.zeros((len(good_without_list), 2), dtype=np.float32)
# points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_without_list]).reshape(-1, 1, 2)
# points2 = np.float32([keypoints2[m.queryIdx].pt for m in good_without_list]).reshape(-1, 1, 2)
for i, match in enumerate(good_without_list):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
sky_points1 = np.ones_like(points1)
sky_points2 = np.ones_like(points2)
ref_map_pix_size = [ref_map.meta['cdelt1'], ref_map.meta['cdelt2']]
for pi in range(points1.shape[0]):
    sky_points1[pi, :] = np.asarray(
        gstt.pix_point_2_sky(x=points1[pi, 0], y=points1[pi, 1], tar_data=im1Gray, tar_map=cur_map, tar_point=True,
                             ref_pix_size=ref_map_pix_size, tar_resized=_tar_resized))
    sky_points2[pi, :] = np.asarray(
        gstt.pix_point_2_sky(x=points2[pi, 0], y=points2[pi, 1], tar_data=im2Gray, tar_map=ref_map,
                             tar_point=False, tar_resized=_tar_resized))
h = cv2.estimateRigidTransform(sky_points1, sky_points2, False)
#print('current matrix is : ', h)
trans = gstt.convert_matrix(h)
aligned_fits = gstt.apply_trans_to_fitsfile(inp_fitsfile=processed_gst_fits, inp_trans=trans)
# plot the aligned image and the reference img in the same FOV
gstt.flick_images(aligned_fits, ref_fits, img_dir=tmp_dir)

