# Linux Command

| Command | Description                                 |
| ------- | ------------------------------------------- |
| du -sh  | 查看文件夹的大小，s表示summrize，仅显示总计 |
| df -h   | 查看磁盘空间的使用情况                      |
| free -h | 查看内存大小                                |
| top     | 查看内存使用情况                            |



### 在mac上挂在ntfs格式的磁盘

```shell
sudo umount /Volumes/UNTITLED
sudo mount -t ntfs -o rw,auto,nobrowse /dev/disk3s1 ~/ntfs-volume
sudo umount ~/ntfs-volume
```

### 压缩相关命令

- tar

  | Command | Description                                                  |
  | ------- | ------------------------------------------------------------ |
  | tar     | `-z(gzip)`      用gzip来压缩/解压缩文件<br/>`-j(bzip2)`     用bzip2来压缩/解压缩文件<br/>`-v(verbose)`   详细报告tar处理的文件信息<br/>`-c(create)`    创建新的档案文件<br/>`-x(extract)`   解压缩文件或目录<br/>`-f(file)`      使用档案文件或设备，这个选项通常是必选的。 |

  举例

  ```shell
  #压缩
  [root@localhost tmp]# tar -zvcf buodo.tar.gz buodo
  [root@localhost tmp]# tar -jvcf buodo.tar.bz2 buodo 
  
  #解压
  [root@localhost tmp]# tar -zvxf buodo.tar.gz 
  [root@localhost tmp]# tar -jvxf buodo.tar.bz2
  ```

- gzip 

  | Command | Description                                                  |
  | ------- | ------------------------------------------------------------ |
  | gzip    | 压缩后的格式为：*.gz <br/>这种压缩方式不能保存原文件；且不能压缩目录 |

  举例

  ```shell
  #压缩
  [root@localhost tmp]# gzip buodo
  [root@localhost tmp]# ls
  buodo.gz
  #解压
  [root@localhost tmp]# gunzip buodo.gz 
  [root@localhost tmp]# ls
  buodo
  ```

- zip

  | Command | Description                                                  |
  | ------- | ------------------------------------------------------------ |
  | zip     | 与gzip相比：1）可以压缩目录； 2）可以保留原文件； <br/>`-r(recursive)`    递归压缩目录内的所有文件和目录 |

  举例

  ```shell
  #压缩和解压文件
  [root@localhost tmp]# zip boduo.zip boduo
  [root@localhost tmp]# unzip boduo.zip
  
  #压缩和解压目录
  [root@localhost tmp]# zip -r Demo.zip Demo
    adding: Demo/ (stored 0%)
    adding: Demo/Test2/ (stored 0%)
    adding: Demo/Test1/ (stored 0%)
    adding: Demo/Test1/test4 (stored 0%)
    adding: Demo/test3 (stored 0%)
  [root@localhost tmp]# unzip Demo.zip 
  Archive:  Demo.zip
     creating: Demo/
     creating: Demo/Test2/
     creating: Demo/Test1/
   extracting: Demo/Test1/test4        
   extracting: Demo/test3  
  ```

- bzip2

  | Command | Description                                                |
  | ------- | ---------------------------------------------------------- |
  | bzip2   | 压缩后的格式：.bz2； <br/>`-k`    产生压缩文件后保留原文件 |

  举例

  ```shell
  #压缩
  [root@localhost tmp]# bzip2 boduo
  [root@localhost tmp]# bzip2 -k boduo
  
  #解压
  [root@localhost tmp]# bunzip2 boduo.bz2 
  ```

- 参考资料：https://blog.csdn.net/capecape/article/details/78548723 



### 文本处理工具

- awk
  - https://www.ruanyifeng.com/blog/2018/11/awk.html



## 阮一峰的linux命令介绍 

https://www.bookstack.cn/read/bash-tutorial/docs-archives-command.md

