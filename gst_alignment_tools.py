import copy
import sunpy.map as smap
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from astropy import units as u
from skimage.transform import rescale, resize
from astropy.coordinates import SkyCoord
import numpy as np
import enhance
from scipy.ndimage import rotate as sci_rotate
import cv2
import imageio


#def halpha_sat_correction_line_by_line(halpha_fits_file):
#    # correct the saturated area in Halpha image line by line based on continuity of the image.
#    # ha_map =
#    return


def make_sub_map(ori_map, fov=None, ref_map=None):
    if not ref_map:
        ref_map = ori_map
    bole = SkyCoord(fov[0][0] * u.arcsec, fov[0][1] * u.arcsec, frame=ref_map.coordinate_frame)
    tori = SkyCoord(fov[1][0] * u.arcsec, fov[1][1] * u.arcsec, frame=ref_map.coordinate_frame)
    sub_map = ori_map.submap(bole, tori)
    return sub_map


def halpha_sat_correction_1st_step(ha_data):
    float_array = copy.deepcopy(ha_data).astype(float)
    nmask = float_array < 0.0
    float_array[nmask] = 32767. * 2. + float_array[nmask]
    return float_array


def fits_header_modifier_gst(filename, tmp_dir):
    tmp_file = os.path.join(tmp_dir, "tmp_" + os.path.basename(filename))
    if os.path.exists(tmp_file):
        os.system('rm {0}'.format(tmp_file))
    os.system('cp {0} {1}'.format(filename, tmp_file))
    with fits.open(tmp_file, mode='update') as cur_hdul:
        chdr = cur_hdul[0].header
        chdr_data = cur_hdul[0].data
        if not 'ALIG_STATE' in list(chdr):
            chdr['CTYPE1'] = 'HPLN-TAN'
            chdr['CTYPE2'] = 'HPLT-TAN'
            chdr['CUNIT1'] = 'arcsec'
            chdr['CUNIT2'] = 'arcsec'
            chdr['DATE-OBS'] = chdr['DATE-OBS'] + 'T' + chdr['TIME-OBS']
            chdr['ALIG_STATE'] = 'FAKE_SDO_FASHION'
            if chdr['INSTRUME'] == 'VIS':
                cur_hdul[0].data= halpha_sat_correction_1st_step(chdr_data)
            cur_hdul.flush()
        else:
            cur_hdul.close()
    cur_map = smap.Map(tmp_file)
    cur_map.meta['crota1'] = -cur_map.meta['crota1']
    cur_map.meta['crota2'] = -cur_map.meta['crota2']
    cur_map.meta['solar_p'] = -cur_map.meta['solar_p']
    rot_map = cur_map.rotate(angle= (cur_map.meta['solar_p'])* u.deg)
    rot_map.meta['naxis1'] = rot_map.data.shape[0]
    rot_map.meta['naxis2'] = rot_map.data.shape[1]
    rot_map.meta['crota1'] = 0.0
    rot_map.meta['crota2'] = 0.0
    rot_map.meta['solar_p'] = 0.0

    #rot_map.meta['solar_p'] = 0.0
    os.system('rm {0}'.format(tmp_file))
    rot_map.save(tmp_file)
    return tmp_file


def compared_arraies(fitsfile, ref_fitsfile, tar_resized=True):
    # pass
    cur_map = smap.Map(fitsfile)
    #rotate_ang = cur_map.meta['solar_p']
    cur_location = [cur_map.meta['crval1'], cur_map.meta['crval2']]
    cur_fov = [[cur_location[0] - cur_map.meta['naxis1'] * cur_map.meta['cdelt1'],
                cur_location[1] - cur_map.meta['naxis2'] * cur_map.meta['cdelt2']],
               [cur_location[0] + cur_map.meta['naxis1'] * cur_map.meta['cdelt1'],
                cur_location[1] + cur_map.meta['naxis2'] * cur_map.meta['cdelt2']]]
    print('Double the FOV of the gst img to draw the reference img, {0}'.format(cur_fov))
    ref_map = smap.Map(ref_fitsfile)
    if ref_map.meta.has_key('detector'):
        if ref_map.meta['detector'] == 'HMI':
            ref_map = ref_map.rotate(angle=180 * u.deg)
    ref_map = make_sub_map(ori_map=ref_map, fov=cur_fov)
    ref_data = ref_map.data
    # resized_ref_data = resize(ref_data,(int(ref_data.shape[0]*ref_map.meta['cdelt1']/cur_map.meta['cdelt1']),int(ref_data.shape[1]*ref_map.meta['cdelt2']/cur_map.meta['cdelt2'])))
    #resized_ref_data = cv2.resize(ref_data,(int(ref_data.shape[0]*ref_map.meta['cdelt1']/cur_map.meta['cdelt1']),int(ref_data.shape[1]*ref_map.meta['cdelt2']/cur_map.meta['cdelt2'])),interpolation = cv2.INTER_AREA)
    # cur_map = cur_map.rotate(angle=rotate_ang * u.deg)
    # cur_map = cur_map.rotate(angle=52.0622398067 * u.deg)
    cur_data = cur_map.data
    fig1, axs = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(6, 6))
    axs.imshow(cur_data,origin='lower')
    #cur_data = sci_rotate(cur_map.data, rotate_ang, reshape=True)
    imit_shape =(int(cur_data.shape[0]*cur_map.meta['cdelt1']/ref_map.meta['cdelt1']),int(cur_data.shape[1]*cur_map.meta['cdelt2']/ref_map.meta['cdelt2']))
    #print(cur_data.shape,imit_shape)
    if tar_resized:
        resized_cur_data = resize(cur_data,imit_shape)
        return (resized_cur_data.astype(float), ref_data.astype(float),cur_map, ref_map)
    else:
        return (cur_data.astype(float), ref_data.astype(float),cur_map,ref_map)

def pix_point_2_sky(x,y,tar_map,tar_data, tar_point=True, ref_pix_size=None, tar_resized=True):
    xc = tar_data.shape[0]/2
    yc = tar_data.shape[1]/2
    if tar_point and tar_resized:
        #print('size', xc, yc)
        x *= ref_pix_size[0]/tar_map.meta['cdelt1']
        y *= ref_pix_size[1]/tar_map.meta['cdelt2']
        xc *=ref_pix_size[0]/tar_map.meta['cdelt1']
        yc *=ref_pix_size[1]/tar_map.meta['cdelt2']
        #pass
    sky_coord = tar_map.pixel_to_world(x * u.pix, y * u.pix)
    sky_coord_rc = tar_map.pixel_to_world(xc * u.pix, yc * u.pix)
    res = [sky_coord.Tx.value - sky_coord_rc.Tx.value, sky_coord.Ty.value - sky_coord_rc.Ty.value]
    if tar_point:
        #print('cur res', res)
        pass
    return res

def contrast_adjustment(inp_img, brightness_range,gamma=1.0):
    cur_img = copy.deepcopy(inp_img).astype(float)
    blk_mask = cur_img < brightness_range[0]
    wht_mask = cur_img > brightness_range[1]
    #adj_mask = ~(blk_mask|wht_mask)
    #print(blk_mask.sum())
    #print(wht_mask.sum())
    #print(adj_mask.sum())
    cur_img[blk_mask] = 0.0
    #cur_img[wht_mask] = 255.0
    cur_img -= brightness_range[0]
    cur_img.clip(0.0)
    cur_img *= 255./(brightness_range[1]-brightness_range[0])
    cur_img[cur_img > 255.] = 255.
    cur_img = 255*np.power(cur_img/255,gamma)
    cur_img = np.around(cur_img)
    cur_img[cur_img>255] = 255
    out_img = cur_img.astype(np.uint8)

    #cur_img[adj_mask] = max(int((cur_img[adj_mask]-brightness_range[0])/(brightness_range[1]-brightness_range[0])*255),int(255))
    return cur_img.astype('uint8')

def convert_matrix(inp_matrix):
    print('costheta.R and sin~~~ are :', inp_matrix[0,0] ,inp_matrix[0,1])
    cscaling = np.sqrt(inp_matrix[0,0]**2 + inp_matrix[0,1]**2)
    ctranslation_xy = [inp_matrix[0,2], inp_matrix[1,2]]
    crotation = np.arccos(inp_matrix[0,0]/cscaling)*(180./np.pi)
    print('Scaling factor is {}, translation is [{},{}] arcsec, rotation is {} degree(unti-clockwised) with respect to the central point of the image'.format(cscaling,ctranslation_xy[0],ctranslation_xy[1],crotation))
    return (cscaling,ctranslation_xy,crotation)

def apply_trans_to_fitsfile(inp_fitsfile, inp_trans):
    file_dir = os.path.dirname(inp_fitsfile)
    outp_file = os.path.join(file_dir, os.path.basename(inp_fitsfile).replace('tmp_','final_'))
    if os.path.exists(outp_file):
        os.system('rm {0}'.format(outp_file))
    #os.system('cp {0} {1}'.format(inp_fitsfile, outp_file))
    cur_map = smap.Map(inp_fitsfile)
    cur_map.meta['CRVAL1'] += inp_trans[1][0]
    cur_map.meta['CRVAL2'] += inp_trans[1][1]
    #cur_map.meta['solar_p'] += (inp_trans[2])
    cur_map.meta['CDELT1'] /= inp_trans[0]
    cur_map.meta['CDELT2'] /= inp_trans[0]
    cur_data = sci_rotate(cur_map.data, inp_trans[2], reshape=False)
    new_map = smap.Map(cur_data, cur_map.meta)
    print('Aligned fits file is saved to ', outp_file)
    new_map.save(outp_file)
    return outp_file

def flicker_images(aligned_fitsfile, ref_fitsfile,img_dir):
    #cfov = [[860.0, -20.0], [940.0, 60.0]]
    cfov = get_fov_of_map(aligned_fitsfile,half=True)
    aligned_map = smap.Map(aligned_fitsfile)
    sub_aligned_map = make_sub_map(aligned_map,fov = cfov)
    ref_map = smap.Map(ref_fitsfile)
    if ref_map.meta.has_key('detector'):
        if ref_map.meta['detector'] == 'HMI':
            ref_map = ref_map.rotate(angle=180 * u.deg)
    sub_ref_map = make_sub_map(ref_map, fov=cfov)
    ali_Gray = np.array(sub_aligned_map.data * 255 / np.nanmax(sub_aligned_map.data), dtype=np.uint8)
    ref_Gray = np.array(sub_ref_map.data * 255 / np.nanmax(sub_ref_map.data), dtype=np.uint8)
    ali_Gray = resize(ali_Gray,ref_Gray.shape)
    img_list = []
    img_list.append(ali_Gray)
    img_list.append(ref_Gray)
    imageio.mimsave(os.path.join(img_dir,'flick.gif'), img_list, fps=4)
    '''
    fig, cax = plt.subplots(figsize=(6, 6))
    im1 = cax.imshow(sub_aligned_map.data, origin='lower')
    plt.savefig(os.path.join(img_dir,'aligned.png'))
    cax.clear()
    im2 = cax.imshow(sub_ref_map.data, origin='lower')
    #print('Am I here?')
    plt.savefig(os.path.join(img_dir,'ref.png'))
    img_list = []
    for ii in range(10):
        img_list.append(resize())
        img_list.append(im2)
    imageio.mimsave(os.path.join(img_dir,'flick.gif'), img_list)
    print('imgs are saved as {0} and {1}'.format(os.path.join(img_dir,'aligned.png'), os.path.join(img_dir,'ref.png')))
    '''
    print('flicker gif is saved to {}'.format(os.path.join(img_dir,'flick.gif')))
    return

def enhance_image(inp_data,f_size,gamma=3.5):
    tmp_data = copy.deepcopy(inp_data).astype(float)
    #new_data = enhance.mgn(data=inp_data, sigma=f_size, gamma=gamma)
    new_data = enhance.mgn(data=inp_data, sigma=f_size)
    return new_data.astype('uint8')

def detail_filter(inp_img):
    KsizeX = 15  # positive ODD
    KsizeY = 14 # positive odd
    out_img = cv2.GaussianBlur(inp_img, (KsizeX, KsizeY), 0)
    return out_img.astype('uint8')

def define_mask(center_mask1,center_mask2, im1, im2):
    mask1 = np.zeros(im1.shape[:2], dtype=np.uint8)
    mask2 = np.zeros(im2.shape[:2], dtype=np.uint8)
    mask1 = np.zeros(im1.shape[:2], dtype=np.uint8)
    x1 = int(im1.shape[0] * (1.0 - center_mask1) / 2)
    x2 = int(im1.shape[0] * (0.5 + center_mask1 / 2))
    y1 = int(im1.shape[1] * (1.0 - center_mask1) / 2)
    y2 = int(im1.shape[1] * (0.5 + center_mask1 / 2))
    cv2.rectangle(mask1, (x1, y1), (x2, y2), (255), thickness=-1)
    mask2 = np.zeros(im2.shape[:2], dtype=np.uint8)
    x1 = int(im2.shape[0] * (1.0 - center_mask2) / 2)
    x2 = int(im2.shape[0] * (0.5 + center_mask2 / 2))
    y1 = int(im2.shape[1] * (1.0 - center_mask2) / 2)
    y2 = int(im2.shape[1] * (0.5 + center_mask2 / 2))
    cv2.rectangle(mask2, (x1, y1), (x2, y2), (255), thickness=-1)
    return mask1, mask2

def get_fov_of_map(inp_fits, half=False):
    inp_map = smap.Map(inp_fits)
    pix_bole_coor = inp_map.pixel_to_world(0 * u.pix, 0 * u.pix)
    pix_tori_coor = inp_map.pixel_to_world((inp_map.data.shape[0]-1) * u.pix, (inp_map.data.shape[1]-1) * u.pix)
    bole_wx = pix_bole_coor.Tx.value
    bole_wy = pix_bole_coor.Ty.value
    tori_wx = pix_tori_coor.Tx.value
    tori_wy = pix_tori_coor.Ty.value
    if half:
        hbole_wx = (bole_wx+tori_wx)/2.0 - (tori_wx-bole_wx)/4.0
        htori_wx = (bole_wx+tori_wx)/2.0 + (tori_wx-bole_wx)/4.0
        hbole_wy = (bole_wy+tori_wy)/2.0 - (tori_wy-bole_wy)/4.0
        htori_wy = (bole_wy+tori_wy)/2.0 + (tori_wy-bole_wy)/4.0
        return [[hbole_wx, hbole_wy], [htori_wx, htori_wy]]
    else:
        return [[bole_wx, bole_wy],[tori_wx, tori_wy]]



