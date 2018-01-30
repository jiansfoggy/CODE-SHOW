# Q1. If macOS Sierra cannot load MySQL Preference Pane

**方法一，从MySQL 5.7.20 DMG包单独安装Preference Pane**

MySQL 5.7.20是没有这个问题，可以从5.7.20的DMG包里单独安装Preference pane。  

> 步骤:  
删除原来的MySQL Preference pane,下载MySQL5.7.20 DMG包。  
下载地址：https://downloads.mysql.com/archives/community/  
安装MySQL5.7.20时，在Installation Type 单独选中Preference Pane安装。

**方法二，命令行启动MySQL**  

>如果不想按第一个方案重新安装MySQL Preference Pane，在bug的讨论里他们有提到使用命令行启动/关闭MySQL服务器。  
启动MySQL  
``sudo launchctl load -F /Library/LaunchDaemons/com.oracle.oss.mysql.mysqld.plist``  
关闭MySQL  
``sudo launchctl unload -F /Library/LaunchDaemons/com.oracle.oss.mysql.mysqld.plist``  
不需要更改以上任何一个代码。

**方法三，Crack it from Terminal**  

If you prefer to reading English, please try this:  
https://devmarketer.io/learn/do-not-install-mysql-macos-sierra-how-to-fix/  



# Q2 If You Forget Your Password for MySQL Roothost Account on You Macbook  

步骤：  
> 苹果->系统偏好设置->最下边点mysql->在弹出页面中关闭mysql服务,  

> 进入终端输入：  
``cd /usr/local/mysql/bin/``  
>
> 回车后，登录管理员权限  
``sudo su``  
>
> 回车后输入以下命令来禁止mysql验证功能  
``./mysqld_safe --skip-grant-tables &``  
>
> 回车后mysql会自动重启，重启好了之后进入mysql workbench 随便创建一个连接，然后用户名填root(注意这里不会验证密码，所以填只要存在的账户就可以,意思是点击root的那个图标会自动进入正常界面)。  
> 再创建一个server administration，选择刚创建的连接。  
> 双击server administration，  
> 左侧点击security，右侧就可以看到所有用户权限表了，这个时候想怎么干都行了。 

以下是部分说明：  
用户权限表中  
Limit Connectivity to Hosts Matching 表示登录地址限制，就是登录时候的ip地址，
‘%'代表任意Adminstrative Roles是权限，如果发现你的root没有管理员权限了，就点这个选项卡全部勾选。  

以下是其他命令:  
``/mysqladmin -u root -p password 123 //更改root用户密码``  
``/mysql -uroot -p //root用户登录mysql``
