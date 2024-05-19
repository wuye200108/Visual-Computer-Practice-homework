# 通过立体匹配获取两张图像的视差图

**算法处理过程：**

1. **图像预处理**：

   - 将输入的左右图像转换为灰度图，这是立体匹配算法通常的要求。

2. **立体匹配算法**：

   - 创建一个 

     ```
     StereoBM
     ```

      对象（Block Matching 算法），用于计算视差图。

     ```
     numDisparities
     ```

      和 

     ```
     blockSize
     ```

      是算法的两个主要参数，可以根据具体应用进行调整。

     - `numDisparities`：视差搜索范围，必须是16的倍数。
     - `blockSize`：匹配块的大小，通常为奇数。

3. **计算视差图**：

   - 使用 `stereo.compute` 方法计算视差图。
   - 使用 `cv2.normalize` 将视差图归一化，以便显示。

4. **显示结果**：

   - 使用 `matplotlib` 库显示左图、右图和视差图。

     **结果**：

     ![image-20240519215505559](C:\Users\14249\AppData\Roaming\Typora\typora-user-images\image-20240519215505559.png)

     