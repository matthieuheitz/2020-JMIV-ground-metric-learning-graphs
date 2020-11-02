# Ground Metric Learning on Graphs

Code for the publication:

Matthieu Heitz, Nicolas Bonneel, David Coeurjolly, Marco Cuturi, and Gabriel Peyré. "Ground Metric Learning on Graphs.", 2020

## Requirements

- ##### Environment

The conda GMLG environment can be set with:
```
conda env create -f environement.yml
conda activate GMLG
```

The fastest version of the code (using a Cholesky solver) requires the Python package `scikit-sparse` and
the CHOLMOD library of the larger SuiteSparse library.
On debian-based systems, you can install the `libcholmod` package.
Otherwise, you can use the LU solver which is twice slower but don't require additional packages (it's in `scipy`).
In that case, specify '--SD_algo LU' when using the scripts.

- ##### C file

In order to have faster convolutions for color transfer, we use a native C implementation.
Compile the `ckernels.c` by simply running `make` in the main directory, which will compile to a `libckernels.so`.


## Starter files:

- `ml_kinterp2.py`: metric learning on a graph that is a 2D grid.
- `ml_color_timelapse.py`: metric learning on a graph that is a 3D grid (for color histograms).

Each of these scripts has arguments, which can be seen with the `-h` option.

## Learning on a 2-D grid

- ##### Create a dataset

In order to create your own histogram sequence (`.npy` files) in 2D, use one of the following scripts:

- `dg_kinterp2.py`: generates a histogram sequence, as an interpolation of k steps between 2 marginals, with a given metric
- `dg_kinterp2_from_images.py`: generates a histogram sequence, from a sequence of images by converting them to grayscale


- ##### Learn a metric

From an existing dataset:
```
# Generate Figure 3
python ml_kinterp2.py data/toydataF3 -o test-F3 --loss_num 1 -L 50 --t_heat 3e-3 --k_heat 100 --metric_regul_ro 0 --metric_regul_ro_lap 0.03 --max_iter 1000 -f 20

# Generate Figure 4
python ml_kinterp2.py data/toydataF4 -o test-F4 --loss_num 4 -L 50 --t_heat 3e-3 --k_heat 100 --metric_regul_ro 0 --metric_regul_ro_lap 0.03 --max_iter 1000 -f 20

# Generate Figure 5
python ml_kinterp2.py data/toydataF5 -o test-F5 --loss_num 2 -L 50 --t_heat 3e-3 --k_heat 100 --metric_regul_ro 1 --metric_regul_ro_lap 1 --max_iter 1000 -f 10

# Generate Figure 6
python ml_kinterp2.py data/toydataF6 -o test-F6 --loss_num 2 -L 50 --t_heat 3e-3 --k_heat 100 --metric_regul_ro 0 --metric_regul_ro_lap 10 --max_iter 1000 -f 20
```

## Learning on a 3-D grid

- ##### Learn a metric

Learn a metric on the `seldovia2` dataset:

`python ml_color_timelapse.py data/seldovia2 -o test-ml-color -n 16 --loss_num 2 -L 50 --t_heat 1e-3 --k_heat 20 --metric_regul_ro 0 --metric_regul_ro_lap 1 --max_iter 500 -f 50`

This script, contrary to the previous one (in 2D) doesn't require `.npy` files, you can directly give it images and
it will compute the color histograms automatically.
If you just want to compute color histograms of images without learning a metric, use the script `dg_compute_color_hist.py`.
For example, to compute color histograms from images with 16 bins per dimension, run the following:
`python dg_compute_color_hist.py "data/seldovia2/*.png" -n 16 -o data/seldovia2/hists16`


- ##### Reuse the metric to create a new sunset sequence

The script `ml_color_transfer.py` performs two tasks:
- `hist_interp`: interpolate between two color histograms with a given metric to get a new histogram sequence
(`interp-hist-*` output files)
- `color_transfer`: transfer the colors of the new histogram sequence on the first input image, to create a 
new image sequence (`interp-??.png` output files)

You can perform only the first task by adding `--only_interp`, or only the second task by providing the histogram sequence
with `--use_existing_interp`. This is useful for example when trying different parameters for the color transfer:
the first task will be the same for all cases, so you don't want to redo it every time. Instead, you first compute
an interpolation and then test different color transfers using that same interpolation.

Examples of how to use this script, to generate images of Figure 18.
```
# Performing both tasks in the same script:
python ml_color_transfer.py "data/country1/*.png" test-ml-color/a-metric-0500.npy -o test-color-transfers \
    --hist_interp input --num_prolong_hi 1 \
    --color_transfer euclid --L_ct 500 --sig_gamma_ct 0.05

python ml_color_transfer.py "data/country1/*.png" test-ml-color/a-metric-0500.npy -o test-color-transfers \
    --hist_interp euclid --num_prolong_hi 1 --gamma_hi 0.001 \
    --color_transfer euclid --L_ct 500 --sig_gamma_ct 0.05

python ml_color_transfer.py "data/country1/*.png" test-ml-color/a-metric-0500.npy -o test-color-transfers \
    --hist_interp linear --num_prolong_hi 1 \
    --color_transfer euclid --L_ct 500 --sig_gamma_ct 0.05
```

Example of how to use this script separately (testing parameters for color transfer)
```
# First, interpolation
python ml_color_transfer.py "data/country1/*.png" test-ml-color/a-metric-0500.npy -o test-color-transfers \
    --only_interp --hist_interp input --num_prolong_hi 1
# output directory is 'test-color-transfers/country1_test-ml-colorI500__hiinput_nhi31_Lhi50_thi1.00e-03_Khi20'

# Then, multiple color transfers
python ml_color_transfer.py "data/country1/*.png" test-ml-color/a-metric-0500.npy -o test-color-transfers \
    --use_existing_interp test-color-transfers/country1_test-ml-colorI500__hiinput_nhi31_Lhi50_thi1.00e-03_Khi20 \
    --color_transfer euclid --L_ct 500 --sig_gamma_ct 0.05

python ml_color_transfer.py "data/country1/*.png" test-ml-color/a-metric-0500.npy -o test-color-transfers \
    --use_existing_interp test-color-transfers/country1_test-ml-colorI500__hiinput_nhi31_Lhi50_thi1.00e-03_Khi20 \
    --color_transfer euclid --L_ct 500 --sig_gamma_ct 0.1

python ml_color_transfer.py "data/country1/*.png" test-ml-color/a-metric-0500.npy -o test-color-transfers \
    --use_existing_interp test-color-transfers/country1_test-ml-colorI500__hiinput_nhi31_Lhi50_thi1.00e-03_Khi20 \
    --color_transfer euclid --L_ct 2000 --sig_gamma_ct 0.05

```

###### Input metric
This script looks needs a `0-parameters.json` file to be in the same directory as the input metric file, in order to obtain
parameters with which the metric was learned (number of Sinkhorn iterations `L`, `t_heat`,`k_heat`, etc.).

###### Input images
In the above examples, we provide 10 `.png` images, but the program only uses the first and the last one.
We could also provide only the first and last image (`"data/country1/video{29,38}.png"`) and get the same result.
The advantage of giving the entire sequence is that the program uses the number of input images to know the number of
interpolation `num_interp` to compute between the first and the last histogram. This is useful for comparison with a ground truth.
When giving only 2 images, the default for `num_interp` is 10.

###### Metric upsampling
Learning a metric in 16^3 is already quite long, but color transfer in 16**3 can present significant quantization errors.
Therefore, we upsample the metric to a higher resolution. `--num_prolong_hi` is the parameter that controls this and
it represents how many times we (roughly) double the metric's resolution. If the input metric is defined on a 3D grid of
size `n`, then the upsampled metric is defined on a grid `n_hi = 2**num_prolong_hi * (n-1) + 1`.

###### Bilateral filtering
As mentioned in the paper, we apply a bilateral filter after the color transfer to reduce color artifacts.
In the output directory, the program writes the final images `interp-??.png`, and the images before applying bilateral
filtering `interp-bbf-??.png`. The bilateral filtering is enabled by default.



## Direct transfer

The script `ml_direct_transfer.py` allows to directly transfer the colors of one or more source images onto one or
more target images.

Examples:
```
# Transfer each image of seldovia2 on the first frame of country1 (last row of Figure 18)
python ml_direct_transfer.py "data/seldovia2/*.png" "data/country1/video29.png" \
    -o test-direct-transfers --n 16 --L 500 --sig_gamma 0.05

# Transfer each image of seldovia2 on each image in country1
python ml_direct_transfer.py "data/seldovia2/*.png" "data/country1/*.png" \
    -o test-direct-transfers --n 16 --L 500 --sig_gamma 0.05

# Transfer each image of seldovia2 on its corresponding image in country1
python ml_direct_transfer.py "data/seldovia2/*.png" "data/country1/*.png" \
    -o test-direct-transfers --n 16 --L 500 --sig_gamma 0.05 --combinations match
```
