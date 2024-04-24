import bdlb
import torch

fs = bdlb.load(benchmark="fishyscapes")
# automatically downloads the dataset
data = fs.get_dataset('LostAndFound')

# test your method with the benchmark metrics
def estimator(image):
    """Assigns a random uncertainty per pixel."""
    uncertainty = torch.random(image.shape[:-1])
    return uncertainty

metrics = fs.evaluate(estimator, data.take(2))
print('My method achieved {:.2f}% AP'.format(100 * metrics['AP']))

# 明天直接用来测试能不能用，里面只有val