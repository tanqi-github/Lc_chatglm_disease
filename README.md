## 快速上手

### 1. 环境配置

# 安装全部依赖
$ pip install -r requirements_lite.txt  

创建名为pwd.py的文件，将其置于python环境的lib文件夹中。

### 2， 启动Neo4j数据库

1、打开终端，使用```shell
$ runas /noprofile
```让普通用户以管理员身份运行命令。
2、进入Neo4j安装目录，执行```shell
$ neo4j.bat console
```以打开Neo4j控制台，直到出现“Started”字样，表示数据库启动成功。

### 3. 一键启动

按照以下命令启动项目

```shell
$ python startup.py -a
```
