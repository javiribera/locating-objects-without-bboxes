### Changelog

#### 1.5.0
 * [b81e40f](../../commit/b81e40f) - __(Javier Ribera)__ + explanation in argparse
 * [298ad6c](../../commit/298ad6c) - __(Javier Ribera)__ added tif as valid extension for imgs
 * [218545e](../../commit/218545e) - __(Javier Ribera)__ random seed for initial split of validation dataset
 * [6110944](../../commit/6110944) - __(Javier Ribera)__ automatically detect XML or CSV, and move plant stuff to separate module
 * [5d0fd32](../../commit/5d0fd32) - __(Javier Ribera)__ new intermediate output w/ crosses on original img
 * [2c03135](../../commit/2c03135) - __(Javier Ribera)__ organize: move some dirs into intermediate dir
 * [54f9c7a](../../commit/54f9c7a) - __(Javier Ribera)__ fix: ignore nan thresholds when plotting kde
 * [b0a1cf3](../../commit/b0a1cf3) - __(Javier Ribera)__ split subrow_grid_location_yx into y and x
 * [2f2c42d](../../commit/2f2c42d) - __(Javier Ribera)__ fix: saving empty CSV
 * [3b8b0f5](../../commit/3b8b0f5) - __(Javier Ribera)__ fix: forgot to create output dir when not evaluating
 * [d7eaf62](../../commit/d7eaf62) - __(Javier Ribera)__ fix: crash when there is not GT (just inference)
 * [c321949](../../commit/c321949) - __(Javier Ribera)__ use XML instead of CSV for inference
 * [abbbeb1](../../commit/abbbeb1) - __(Javier Ribera)__ fix: build dictionary with ints instead of tensors
 * [8853d76](../../commit/8853d76) - __(Javier Ribera)__ update to pytorch 1.0.0
 * [04a582a](../../commit/04a582a) - __(Javier Ribera)__ fix: convert str to float when chekcing argparse
 * [c87cd4b](../../commit/c87cd4b) - __(Javier Ribera)__ force some arguments to be strictly positive
 * [de05268](../../commit/de05268) - __(Javier Ribera)__ argparse: dont show default if empty str
 * [c2b27c7](../../commit/c2b27c7) - __(Javier Ribera)__ mention which args are mandatory, and which optional
 * [4ac1f58](../../commit/4ac1f58) - __(Javier Ribera)__ update visdom to newer version
 * [4243c12](../../commit/4243c12) - __(Javier Ribera)__ move to XML, allow partial GT in XML, and 80-20 autosplit w/ XML
 * [c6d19ff](../../commit/c6d19ff) - __(Javier Ribera)__ fix visdom not connecting w/ workaround
 * [c85cbfe](../../commit/c85cbfe) - __(Javier Ribera)__ make visdom optional, port modifiable, and print visdom hostname
 * [7692159](../../commit/7692159) - __(Javier Ribera)__ fix (estimated map must be 1D)
 * [50cf1bc](../../commit/50cf1bc) - __(Javier Ribera)__ mention # images for training/validation
 * [2fee7e6](../../commit/2fee7e6) - __(Javier Ribera)__ go back to UNet
 * [502daa8](../../commit/502daa8) - __(Javier Ribera)__ full EM
 * [5aabb5a](../../commit/5aabb5a) - __(Javier Ribera)__ new option to replace optimizer

#### 1.4.0
 * [86cc171](../../commit/86cc171) - __(Javier Ribera)__ removed non-used models
 * [b8c6ba1](../../commit/b8c6ba1) - __(Javier Ribera)__ read from XML v0.4
 * [101cfa3](../../commit/101cfa3) - __(Javier Ribera)__ fix: get_image_size in results script
 * [70cd619](../../commit/70cd619) - __(Javier Ribera)__ max_ahd=diagonal in metrics_from_results script
 * [2de70e3](../../commit/2de70e3) - __(Javier Ribera)__ new script to make metrics CSV from results CSV
 * [c5b487c](../../commit/c5b487c) - __(Javier Ribera)__ trying CSRNet
 * [a52f835](../../commit/a52f835) - __(Javier Ribera)__ merge BMM density and threshold in same plot
 * [fd3afd8](../../commit/fd3afd8) - __(Javier Ribera)__ cap BMM plots at density=8 instead of 50
 * [40f0094](../../commit/40f0094) - __(Javier Ribera)__ heatmap on top of original image during inference
 * [e5c7498](../../commit/e5c7498) - __(Javier Ribera)__ force R=1 when object count is known
 * [21d4462](../../commit/21d4462) - __(Javier Ribera)__ use default types of fixed # of objects
 * [bed824d](../../commit/bed824d) - __(Javier Ribera)__ nicer plots
 * [bf9d0cb](../../commit/bf9d0cb) - __(Javier Ribera)__ allow [] in taus in arguments
 * [a1c9209](../../commit/a1c9209) - __(Javier Ribera)__ move legend up in the plots
 * [5562a13](../../commit/5562a13) - __(Javier Ribera)__ go back use SmoothL1Loss for regression
 * [e67753f](../../commit/e67753f) - __(Javier Ribera)__ new script to draw circles on top of images of a dataset
 * [7ce6002](../../commit/7ce6002) - __(Javier Ribera)__ remove unused variables
 * [91006d0](../../commit/91006d0) - __(Javier Ribera)__ fix: p=-1 by default to prevent NaNs
 * [4f5b7a4](../../commit/4f5b7a4) - __(Javier Ribera)__ fix: prevent NaNs by not using activ in last layers of U-Net
 * [4645147](../../commit/4645147) - __(Javier Ribera)__ fix: use seed in random transforms too
 * [9890290](../../commit/9890290) - __(Javier Ribera)__ paint red haircrosses on estimated pts during validation
 * [9747046](../../commit/9747046) - __(Javier Ribera)__ go back to use Otsu thresholding
 * [386e4f3](../../commit/386e4f3) - __(Javier Ribera)__ use seed when shuffling dataset at the beginning
 * [d79a1d7](../../commit/d79a1d7) - __(Javier Ribera)__ also paint white circles on top of training labels
 * [6504a7c](../../commit/6504a7c) - __(Javier Ribera)__ paint red circles on top of estimated points and send to Visdom
 * [61e70d4](../../commit/61e70d4) - __(Javier Ribera)__ send heatmaps on top of original to visdom
 * [6a57704](../../commit/6a57704) - __(Javier Ribera)__ shuffle dataset at the beginning
 * [166af89](../../commit/166af89) - __(Javier Ribera)__ fix: use mean of second Beta in BMM thresholding
 * [034b96f](../../commit/034b96f) - __(Javier Ribera)__ use BMM instead of Otsu thresholding during validation
 * [6812ff3](../../commit/6812ff3) - __(Javier Ribera)__ fix: tau is a float, not an int
 * [c2fdeed](../../commit/c2fdeed) - __(Javier Ribera)__ fix: skip validation every val_freq times
 * [362ffca](../../commit/362ffca) - __(Javier Ribera)__ new script to find optimal learning rate (order of magnitude)
 * [16b5d94](../../commit/16b5d94) - __(Javier Ribera)__ new option --force to overwrite output testing results
 * [4464750](../../commit/4464750) - __(Javier Ribera)__ track and log running average loss
 * [d988c42](../../commit/d988c42) - __(Javier Ribera)__ also send avg loss to visdom
 * [335cc16](../../commit/335cc16) - __(Javier Ribera)__ fix: logger not accepting numbers
 * [0f54e01](../../commit/0f54e01) - __(Javier Ribera)__ fix: no need for model.train() for every batch (now faster)
 * [a68072d](../../commit/a68072d) - __(Javier Ribera)__ added copyright info of BMM fitting
 * [c1aab76](../../commit/c1aab76) - __(Javier Ribera)__ nicer plots
 * [d2e9490](../../commit/d2e9490) - __(Javier Ribera)__ fix: multiple radii as input
 * [a393e06](../../commit/a393e06) - __(Javier Ribera)__ use Beta Mixture Model-based thresholding with tau=-2
 * [a6db2b8](../../commit/a6db2b8) - __(Javier Ribera)__ fit spherical GMM so it runs faster
 * [3a37888](../../commit/3a37888) - __(Javier Ribera)__ subsample mask points randomly so GMM fitting is faster
 * [4c65e51](../../commit/4c65e51) - __(Javier Ribera)__ fix: validation was not being done
 * [ccc414b](../../commit/ccc414b) - __(Javier Ribera)__ label otsu thresholding in metric plots
 * [01b01f5](../../commit/01b01f5) - __(Javier Ribera)__ fix: take only some radii if too many to plot
 * [0147511](../../commit/0147511) - __(Javier Ribera)__ rename variable windows->window_ids
 * [1276bee](../../commit/1276bee) - __(Javier Ribera)__ fix: visdom not show validation because reusing window_ids
 * [becbfd0](../../commit/becbfd0) - __(Javier Ribera)__ clearer print msg
 * [413f526](../../commit/413f526) - __(Javier Ribera)__ fix: show always same # of decimals of tau
 * [b7091d1](../../commit/b7091d1) - __(Javier Ribera)__ fix: scale estimated map before saving as img
 * [0c2bde3](../../commit/0c2bde3) - __(Javier Ribera)__ fix: multiple taus in argparse
 * [75d0bdc](../../commit/75d0bdc) - __(Javier Ribera)__ use AMSgrad, the "convergence fix" for Adam
 * [22dd9b5](../../commit/22dd9b5) - __(Javier Ribera)__ always printing losses should not be there besides for debugging
 * [02af554](../../commit/02af554) - __(Javier Ribera)__ p=9 by default
 * [7247aa3](../../commit/7247aa3) - __(Javier Ribera)__ new 2nd term in the cost function
 * [b0db956](../../commit/b0db956) - __(Javier Ribera)__ clarify README instructions
 * [ed72736](../../commit/ed72736) - __(Javier Ribera)__ no need to save checkpoint at the end of each epoch if no validation
 * [38f762c](../../commit/38f762c) - __(Javier Ribera)__ smaller scatter markers
 * [ea968e7](../../commit/ea968e7) - __(Javier Ribera)__ 50 taus instead of 100
 * [288288b](../../commit/288288b) - __(Javier Ribera)__ added copyright notices to all files
 * [9e1b82b](../../commit/9e1b82b) - __(Javier Ribera)__ script is now part of the package
 * [0ae815c](../../commit/0ae815c) - __(Javier Ribera)__ fix doc: setup.py said the package was inference-only
 * [8840a16](../../commit/8840a16) - __(Javier Ribera)__ able to send visualiz to remote Visdom server
 * [f5b1490](../../commit/f5b1490) - __(Javier Ribera)__ conda environment: missing ballpark dependency
 * [e1d8af6](../../commit/e1d8af6) - __(Javier Ribera)__ wrap clustering into a function
 * [018157c](../../commit/018157c) - __(Javier Ribera)__ fix: undo accidental removal of line in commit d78ecf6
 * [ecfd007](../../commit/ecfd007) - __(Javier Ribera)__ max AHD=max dist instead of inf during training
 * [d78ecf6](../../commit/d78ecf6) - __(Javier Ribera)__ encapsulate thresholding into a function
 * [a726245](../../commit/a726245) - __(Javier Ribera)__ corner case: avoid crash when tau = -1 only
 * [570dc74](../../commit/570dc74) - __(Javier Ribera)__ better var names, torch0.4 optimizations, and use centroids_wrt_orig
 * [61ad18b](../../commit/61ad18b) - __(Javier Ribera)__ mark Otsu result separate in the metric plots
 * [dedb81b](../../commit/dedb81b) - __(Javier Ribera)__ fix: not reporting MAE
 * [83b3628](../../commit/83b3628) - __(Javier Ribera)__ use item() from pythorch0.4
 * [f7d53c0](../../commit/f7d53c0) - __(Javier Ribera)__ forgot Normalizer class
 * [43fc661](../../commit/43fc661) - __(Javier Ribera)__ use Otsu thresholding in testing
 * [65bde7c](../../commit/65bde7c) - __(Javier Ribera)__ show # params during testing
 * [23b4c41](../../commit/23b4c41) - __(Javier Ribera)__ normalize centroids in a new class
 * [63f743f](../../commit/63f743f) - __(Javier Ribera)__ show number of parameters
 * [58a377c](../../commit/58a377c) - __(Javier Ribera)__ fix corner case when img has no object at all
 * [77a2f88](../../commit/77a2f88) - __(Javier Ribera)__ use non-default port number in visdom
 * [5bab050](../../commit/5bab050) - __(Javier Ribera)__ one painted output per threshold value
 * [082c07a](../../commit/082c07a) - __(Javier Ribera)__ tau=1 => otsu thresholding
 * [507642c](../../commit/507642c) - __(Javier Ribera)__ change default visdom environment name
 * [ec0d9b2](../../commit/ec0d9b2) - __(Javier Ribera)__ also show r and R^2 on validation in visdom
 * [ad7d18e](../../commit/ad7d18e) - __(Javier Ribera)__ fix: dimension mismatch
 * [8f9824b](../../commit/8f9824b) - __(Javier Ribera)__ avoid matplotlib warning
 * [aab2727](../../commit/aab2727) - __(Javier Ribera)__ corner case in WHD when there are no GT pts
 * [6349981](../../commit/6349981) - __(Javier Ribera)__ use default types
 * [311225d](../../commit/311225d) - __(Javier Ribera)__ seed is 0 now
 * [15030d7](../../commit/15030d7) - __(Javier Ribera)__ also scale image during testing
 * [c1d82c9](../../commit/c1d82c9) - __(Javier Ribera)__ fix matplotlib warning by closing unused figures
 * [113cc3f](../../commit/113cc3f) - __(Javier Ribera)__ also compute r and R2 metrics
 * [15b03a7](../../commit/15b03a7) - __(Javier Ribera)__ plotting metrics inside py package, and plot when locating too
 * [4de5506](../../commit/4de5506) - __(Javier Ribera)__ new script to plot multiple metrics from the results of a CSV
 * [7be5a71](../../commit/7be5a71) - __(Javier Ribera)__ use otsu thresholding
 * [4218727](../../commit/4218727) - __(Javier Ribera)__ fix CSV dataset loading
 * [0f3a3dc](../../commit/0f3a3dc) - __(Javier Ribera)__ put additional info in output CSV

#### 1.3.1
 * [9969e3b](../../commit/9969e3b) - __(Javier Ribera)__ v1.3.1
 * [6937c18](../../commit/6937c18) - __(Javier Ribera)__ no more wheels
 * [988a5b1](../../commit/988a5b1) - __(Javier Ribera)__ allow empty checkpoint directory
 * [191fb3c](../../commit/191fb3c) - __(Javier Ribera)__ show error when dataset directory is empty
 * [c2c05c5](../../commit/c2c05c5) - __(Javier Ribera)__ object-location -> object-locator

#### 1.3.0
 * [b53d485](../../commit/b53d485) - __(Javier Ribera)__ fix environment
 * [1156a67](../../commit/1156a67) - __(Javier Ribera)__ environment.yml for Windows
 * [c0d5490](../../commit/c0d5490) - __(Javier Ribera)__ change deprecated class to remove warning
 * [859be9d](../../commit/859be9d) - __(Javier Ribera)__ better argparse help

