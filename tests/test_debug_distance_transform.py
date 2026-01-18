import numpy as np
import matplotlib.pyplot as plt

def test_distance_transform_values():
    """查看 debug/debug_distance_transform.npy 的值"""

    # 加载数据
    data = np.load('debug/debug_distance_transform.npy')

    print('=== Basic Statistics ===')
    print(f'Shape: {data.shape}')
    print(f'Data type: {data.dtype}')
    print(f'Min value: {data.min()}')
    print(f'Max value: {data.max()}')
    print(f'Mean value: {data.mean()}')
    print(f'Median value: {np.median(data)}')
    print(f'Std dev: {data.std()}')

    print('\n=== Non-zero Statistics ===')
    non_zero_mask = data > 0
    non_zero_count = np.sum(non_zero_mask)
    print(f'Non-zero count: {non_zero_count} / {data.size} ({100*non_zero_count/data.size:.2f}%)')

    if non_zero_count > 0:
        non_zero_values = data[non_zero_mask]
        print(f'Non-zero min: {non_zero_values.min()}')
        print(f'Non-zero max: {non_zero_values.max()}')
        print(f'Non-zero mean: {non_zero_values.mean()}')
        print(f'Non-zero median: {np.median(non_zero_values)}')

    print('\n=== Value Distribution (Percentiles) ===')
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    values = np.percentile(data, percentiles)
    for p, v in zip(percentiles, values):
        print(f'{p}th percentile: {v:.4f}')

    print('\n=== Sample of Non-Zero Values ===')
    if non_zero_count > 0:
        sample_size = min(50, non_zero_count)
        # 获取非零值位置
        non_zero_positions = np.argwhere(non_zero_mask)
        sample_indices = np.random.choice(len(non_zero_positions), sample_size, replace=False)

        print(f'Random sample of {sample_size} non-zero values with positions:')
        for idx in sample_indices[:20]:
            y, x = non_zero_positions[idx]
            print(f'  Position ({y:4d}, {x:4d}): {data[y, x]:.4f}')

    # 可视化
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 原始距离变换图
        im1 = axes[0, 0].imshow(data, cmap='jet')
        axes[0, 0].set_title('Distance Transform')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0, 0])

        # 2. 二值化：非零区域
        binary = (data > 0).astype(np.uint8)
        axes[0, 1].imshow(binary, cmap='gray')
        axes[0, 1].set_title('Non-zero Regions (Binary)')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Y')

        # 3. 直方图
        if non_zero_count > 0:
            axes[1, 0].hist(non_zero_values.flatten(), bins=50, edgecolor='black')
            axes[1, 0].set_title('Histogram of Non-Zero Values')
            axes[1, 0].set_xlabel('Distance Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)

        # 4. 距离值的累积分布
        if non_zero_count > 0:
            sorted_values = np.sort(non_zero_values.flatten())
            cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            axes[1, 1].plot(sorted_values, cumulative)
            axes[1, 1].set_title('Cumulative Distribution of Non-Zero Values')
            axes[1, 1].set_xlabel('Distance Value')
            axes[1, 1].set_ylabel('Cumulative Probability')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图像
        output_path = 'tests/testcases/debug_distance_transform_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'\n可视化结果已保存到: {output_path}')
        plt.close()

    except Exception as e:
        print(f'\n无法创建可视化: {e}')


if __name__ == '__main__':
    test_distance_transform_values()
