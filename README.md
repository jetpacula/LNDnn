# LNDnn

Purpose of the project- gain ability to spot mineral deficiency of the soil on aerial RGB imagery

At first dataset is downloaded from public S3.
Then with custom dataloader class I process source images to RGB objects, and masks to 1-layered image and convert it to tensors afterwards.
After defining loss, optimizers and model itself the training occurs. 
Current maximum IoU metric is ~60%.

After gaining results with DeepLab network I have used static quantization to reduce the network size.

#  Future work
1. To use all three source images per field, combining them into one tensor.
Now torch "stack" method combines them into a tensor with 3 subtensors, which is problematic to use during training.

2. To split dataset into groups of fields (containing 0-10% minerals and >10% minerals) and use bagging to improve IoU results
