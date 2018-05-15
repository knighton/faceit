# faceit

A deep convolutional face detector in PyTorch.

![Some demos](https://github.com/knighton/faceit/raw/master/umdfaces.png)

## 1. Overview

Faceit is a joint model traind on UMDFaces that takes color 128x128 images and predicts:
1. whether there is a face
2. gender
3. pose (yaw, pitch, roll)
4. face bounding box
5. eyes.

![More demos](https://github.com/knighton/faceit/raw/master/drivers.png)

## 2. Architecture

The model feeds the input image through a long central trunk of blocks with skip connections to the five branches, one per output.  This "trunk" squashes the images with (sometimes strided) convolutions, then flattens and does some affine transformations, resulting in an embedding vector.

```
        self.features = nn.Sequential(
            conv_bn_pool(3, k),

            IsoConvBlock(k),
            IsoConvBlock(k),
            IsoConvBlock(k),

            ReduceBlock(k),

            IsoConvBlock(k),
            IsoConvBlock(k),
            IsoConvBlock(k),

            ReduceBlock(k),

            IsoConvBlock(k),
            IsoConvBlock(k),
            IsoConvBlock(k),

            ReduceBlock(k),

            IsoConvBlock(k),
            IsoConvBlock(k),
            IsoConvBlock(k),

            ReduceBlock(k),

            IsoConvBlock(k),
            IsoConvBlock(k),
            IsoConvBlock(k),

            ReduceBlock(k),

            IsoConvBlock(k),
            IsoConvBlock(k),
            IsoConvBlock(k),

            ReduceBlock(k),

            Flatten(),

            IsoDenseBlock(k),
            IsoDenseBlock(k),
            IsoDenseBlock(k),

            IsoDenseBlock(k),
            IsoDenseBlock(k),
            IsoDenseBlock(k),

            IsoDenseBlock(k),
            IsoDenseBlock(k),
            IsoDenseBlock(k),

            IsoDenseBlock(k),
            IsoDenseBlock(k),
            IsoDenseBlock(k),
        )

        self.is_face = nn.Sequential(
            IsoDenseBlock(k),
            IsoDenseBlock(k),
            IsoDenseBlock(k),

            nn.Linear(k, 1),
            nn.Sigmoid(),
        )

        self.is_male = nn.Sequential(
            IsoDenseBlock(k),
            IsoDenseBlock(k),
            IsoDenseBlock(k),

            nn.Linear(k, 1),
            nn.Sigmoid(),
        )

        self.get_pose = nn.Sequential(
            IsoDenseBlock(k),
            IsoDenseBlock(k),
            IsoDenseBlock(k),

            nn.Linear(k, 3),
            Degrees(),
        )

        self.get_face_bbox = nn.Sequential(
            IsoDenseBlock(k),
            IsoDenseBlock(k),
            IsoDenseBlock(k),

            IsoDenseBlock(k),
            IsoDenseBlock(k),
            IsoDenseBlock(k),

            nn.Linear(k, 4),
        )

        self.get_keypoints = nn.Sequential(
            IsoDenseBlock(k),
            IsoDenseBlock(k),
            IsoDenseBlock(k),

            IsoDenseBlock(k),
            IsoDenseBlock(k),
            IsoDenseBlock(k),

            nn.Linear(k, 4),
        )
```

## 3. Blocks

A block is collection of pathways paired with multipliers/"switches"/"gates" that are learned.  The isomorphic blocks have skip connections and the reduce blocks have pooling pathways, which allows you to stack them as deep as you want.  What I think is neat about this design is as follows: the pathways have different levels of complexity.  You can monitor the switches during training to see how much it has to rely on the slower-to-learn complex pathways vs the skip connections. This allows you to tune architecture depth.  This could be done automatically to grow a network from scratch as the switches tell you it is begging for additional capacity.  Furthermore, you could use switch monitoring to go the other direction: retraining a network with a block removed at a time, with precise information about how capacity is holding up, to speed performance in tight ennvironments.  I really think this direction should be explored <<bat signal>> but no time for now.

Three kinds of blocks:

Convolutional isomorphic (initialized with weights [1, 0, 0]):
* Skip connection
* Convolution -- conv2d, batch norm, relu, optionally dropout
* Gated convolution (a primitive form of attention -- see Gated Convolutional Networks paper)

Fully connected isomorphic (initialized with weights [1, 0, 0, 0]):
* Skip connection
* Affine transformation -- linear, batch norm, relu, dropout
* Gated affine (affine equivalent of gated conv)

Convolutional reduce (initialized with weights [0.5, 0.5, 0, 0]):
* Average pooling
* Max pooling (in theory you might worry about overfitting, but in practice I removed dropout)
* Strided convolution
* Gated strided convolution

I like this design because you can invent new pathways, and see very quickly how much the model relies on it, or not.

Other pathways I experimented with:
* Fully-connected block: Two outputs; multiply their sqrt of n + 1 into one output.  The scaling is to fight nans/gradient explosion.  More general/powerful than sigmoid multiply gating?  Perhaps having to have two outputs coincide in such a manner is akin to dropout and may have similar properties?
* Iso conv block: do global max/average pooling, then affine transform that (dimensionality: num filters x num filters).  The model loved this information, but did not converge faster.  Dimensionality may have been too low, further experiments needed.
* ...

## 4. Losses

Losses were selected and balanced empirically.

1. Whether face: binary cross-entropy / 4
2. Gender: binary cross-entropy / 4
3. Pose: mean absolute error / 32 per float
4. Face bounding box: clamp the predicted coordinates to avoid gradient explosion, then take the Euclidean distance / 32.  Was originally fancier.
5. Eyes: see #4.

Gender prediction accuracy as well as average bounding box and eye distance in pixels are also collected during training.  It gets down to about mean 3 pixels for eyes and mean 4 pixels for face bounding box on validation pretty quickly on GPU.  I was satisfied with that performance looking at demo images during training and don't have the resources to really experiment.  Because of the weird stacked gating, it's easier to improve on that by [adding more layers](https://qph.fs.quoracdn.net/main-qimg-2b1f074e9659128d405e3d87a13ae308-c).

## 5. Dataset

The excellent UMDFaces dataset was used, without data augmentation due to time.  Experimenting on a harder set of videos of people driving cars, I leawrned to randomly darken input images during training to ameliorate performance at nighttime.  The effects of this were not quantified due to time constraints.  To improve performance with various occlusions such as glasses, I also tried retraining with the Specs on Faces face detection dataset, although it's a bit small.
