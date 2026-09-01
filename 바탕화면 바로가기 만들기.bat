@echo off
chcp 65001 > nul
powershell -NoProfile -Command ^
  "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\MAENG_Image_Sync.lnk');" ^
  "$s.TargetPath='%~dp0MAENG_Image_Sync 실행.bat';" ^
  "$s.WorkingDirectory='%~dp0';" ^
  "$s.IconLocation='%~dp0MAENG_Image_Sync.ico,0';" ^
  "$s.Description='초고속 카메라 조건 비교 앱';" ^
  "$s.Save(); Write-Host '바로가기 생성 완료 (MAENGLAB 아이콘)'"
pause
