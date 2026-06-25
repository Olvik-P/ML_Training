
```powershell
wsl --import Ubuntu-22.04 D:\WSL\Ubuntu .\ubuntu-22.04.5-wsl-amd64.wsl --version 2
```

```powershell
wsl --list --verbose
  NAME              STATE           VERSION
* docker-desktop    Stopped         2
  Ubuntu-22.04      Stopped         2
```

Шаг 4. Сделайте новый дистрибутив основным

```powershell
wsl --set-default Ubuntu-22.04
Шаг 5. Запустите Ubuntu
```

**3. Запустить Ubuntu**

```powershell
wsl -d Ubuntu-22.04
adduser myuser
 adduser myuser sudo
```