import my_filter_function as mff
import my_plot_functions as mpf

def experimant_image_mask_x5(windows_name, origin_PQ, origin_FFT2, filter_type, pass_type, arr_filter_radius, arr_filt_ord=[2,2,2,2,2]):
    x = 5
    filter_function = [None] * x
    images = [None] * x * 2
    titles = [None] * x * 2
    for cnt in range(x):
        filter_function[cnt] = mff.get_filter_function(origin_PQ, filter_type, pass_type, arr_filter_radius[cnt], arr_filt_ord[cnt])
        images[cnt] = mff.get_filter_image(origin_PQ, origin_FFT2, filter_function[cnt])
        images[cnt + x] = mff.get_filter_mask(filter_function[cnt])
        titles[cnt] = f'Image {pass_type} {filter_type} D0 = {arr_filter_radius[cnt]}'
        titles[cnt + x] = f'Mask {pass_type} {filter_type} D0 = {arr_filter_radius[cnt]}'
    mpf.show_images_group_2x5(images, titles, windows_name)
   

def experimant_image_x10(windows_name, origin_PQ, origin_FFT2, filter_type, pass_type, arr_filter_radius, arr_filt_ord=[2]*10):
    x = 10
    filter_function = [None] * x
    images = [None] * x
    titles = [None] * x
    for cnt in range(x):
        filter_function[cnt] = mff.get_filter_function(origin_PQ, filter_type, pass_type, arr_filter_radius[cnt], arr_filt_ord[cnt])
        images[cnt] = mff.get_filter_image(origin_PQ, origin_FFT2, filter_function[cnt])
        if filter_type == 'btw':
            titles[cnt] = f'D0 = {arr_filter_radius[cnt]}, n = {arr_filt_ord[cnt]}'
        else:
            titles[cnt] = f'D0 = {arr_filter_radius[cnt]}'
    mpf.show_images_group(2, x // 2, images, titles, windows_name)
   
   
