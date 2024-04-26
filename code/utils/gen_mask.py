import numpy as np
from scipy import ndimage


def gen_disjoint_mask(k_list, disjoint_n, dim_h, dim_w):
    while True:
        masks = []
        for cell_k in k_list:
            cell_num = (dim_h // cell_k) * (dim_w // cell_k)  # 计算每个单元的数量
            indicies = np.arange(cell_num)  # 创建索引数组
            np.random.shuffle(indicies)   # 随机打乱索引

            bound = len(indicies) // disjoint_n  # 确定每个区间的边界
            for idx_d in range(disjoint_n):
                mask_i = np.ones(cell_num)  # 初始化遮罩为全1，创建了一个一维数组，其中每个元素都是1。这个数组的长度（即元素的数量）是cell_num。
                split_s = bound * idx_d  # 计算分割的开始索引
                split_e = bound * (idx_d+1)  # 计算分割的结束索引
                if (idx_d >= disjoint_n-1):
                    disjoint_i = indicies[split_s:]  # 最后一个区间取剩余所有
                else:
                    disjoint_i = indicies[split_s:split_e]  # 根据索引分割

                # 将选中的索引在遮罩中设置为0
                mask_i[disjoint_i] = 0
                mask_i = mask_i.reshape(dim_h // cell_k, dim_w // cell_k)  # 调整遮罩形状
                mask_i = ndimage.zoom(mask_i, zoom=cell_k, order=0)  # 放大遮罩以匹配原始图像大小
                masks.append(mask_i)

        # for mask in masks:
        #     print(mask.shape)

        masks = np.array(masks, dtype=np.float32)  # 转换为NumPy数组
        yield masks



