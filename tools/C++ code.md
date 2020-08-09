# C++ code

- 传给函数一个指针，并要在函数内部指向一个创建数组，函数退出时，要将数组的值复制给一段指针所指的空间。否则数组会被析构。

  ```c++
  hi_img->rgb_data = (unsigned char*)malloc(cv_img_roi.rows * cv_img_roi.cols * 3);
  memcpy(hi_img->rgb_data, cv_img_roi.data, cv_img_roi.rows * cv_img_roi.cols * 3);
  ```

  

- 输出指针所指的一段空间的内容

  ```c++
  unsigned char* f1 = (unsigned char *) (hi_img1.rgb_data) ;
  for (int i=0;i<20;i++){
      std::cout<< int(f1[i]) << " ";
  }
  std::cout<< std::endl;
  ```

  