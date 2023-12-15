import os
import cv2
import my_filter_function as mff
import my_plot_functions as mpf
import my_lab_functions as mlf

os.system("cls")

path_image_1 = 'Pic1.jpg'
path_image_2 = 'Pic2.jpg'

origin_image, origin_PQ, origin_FFT2 = mff.get_originals_param(path_image_1)

mlf.experimant_image_x10('1. Низькочастотна фільтрація ідеальним фільтром.', origin_PQ, origin_FFT2, 'ideal', 'lp', range(10, 201, 10))
mlf.experimant_image_x10('2.1 Низькочастотна фільтрація фільтром Баттерворта.', origin_PQ, origin_FFT2, 'btw', 'lp', range(10, 201, 10))
mlf.experimant_image_x10('2.2 Низькочастотна фільтрація фільтром Баттерворта.', origin_PQ, origin_FFT2, 'btw', 'lp', [10]*10, range(1,11))
mlf.experimant_image_x10('3. Низькочастотна фільтрація фільтром Гауса.', origin_PQ, origin_FFT2, 'gaussian', 'lp', range(10, 201, 10))
mlf.experimant_image_x10('4. Високочастотна фільтрація ідеальним фільтром.', origin_PQ, origin_FFT2, 'ideal', 'hp', range(11, 1, -1))
mlf.experimant_image_x10('5.1 Високочастотна фільтрація фільтром Баттерворта.', origin_PQ, origin_FFT2, 'btw', 'hp', range(1, 110, 10))
mlf.experimant_image_x10('5.2 Високочастотна фільтрація фільтром Баттерворта.', origin_PQ, origin_FFT2, 'btw', 'hp', [10]*10, range(1, 110, 10))
mlf.experimant_image_x10('6. Високочастотна фільтрація фільтром Гауса.', origin_PQ, origin_FFT2, 'gaussian', 'hp', range(1, 110, 10))


images = [None] * 2
images[0] = origin_image
images[1] = mff.get_filter_image(origin_PQ, origin_FFT2, mff.get_filter_function(origin_PQ, 'laplacian', 'hp'))

plots = [None] * 2
plots[0] = mff.get_histograme(images[0])
plots[1] = mff.get_histograme(images[1])

titles = [None] * 4
titles[0] = 'Оригінал'
titles[1] = 'Відфільтроване зображення'
titles[2] = 'Гістограма оригіналу'
titles[3] = 'Гістограма відфільтрованого зображення'

mpf.show_images_x2_and_plot_x2(images, plots, titles, '7. Високочастотна фільтрація лапласіаном.')




origin_image, origin_PQ, origin_FFT2 = mff.get_originals_param(path_image_2)

images = [None] * 3
images[0] = cv2.cvtColor(cv2.imread(path_image_2), cv2.COLOR_RGB2GRAY)
images[1] = mff.get_equalized_image(path_image_2)
images[2] = mff.get_filter_image(origin_PQ, origin_FFT2, mff.get_filter_function(origin_PQ, 'btw', 'hp', 2, 1))

plots = [None] * 3
plots[0] = mff.get_histograme(images[0])
plots[1] = mff.get_histograme(images[1])
plots[2] = mff.get_histograme(images[2])

titles = [None] * 6
titles[0] = 'Оригінал'
titles[1] = 'Еквалізація'
titles[2] = 'hp btw D0 = 2 n = 1'
titles[3] = 'Гістограма оригіналу'
titles[4] = 'Гістограма еквалізації'
titles[5] = 'Гістограма частотного методу'

mpf.show_images_x3_and_plot_x3(images, plots, titles, '8. Порівняння фільтрації в просторовій та частотній областях.')