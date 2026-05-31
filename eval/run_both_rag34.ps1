$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location $PSScriptRoot\..
$py = ".\.venv\Scripts\python.exe"
$log = "eval\results\rag34_run.log"

function Set-Model($name) {
  $c = Get-Content .env -Raw
  $c = $c -replace '(?m)^MODEL_NAME=.*$', "MODEL_NAME=$name"
  Set-Content .env $c -NoNewline -Encoding utf8
}

"=== START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $log -Encoding utf8

# Chay tuan tu: moi model -> set .env, deploy (nap model + corpus moi), generate+judge.
$models = @("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "Qwen/Qwen3.5-9B")
foreach ($m in $models) {
  "`n##### MODEL = $m #####" | Out-File $log -Append -Encoding utf8
  Set-Model $m
  "--- deploy ---" | Out-File $log -Append -Encoding utf8
  & $py -m modal deploy modal_app.py *>> $log
  "--- run rag_live (generate+judge) ---" | Out-File $log -Append -Encoding utf8
  & $py eval\run_eval_judge.py --set rag_live *>> $log
}

"`n=== DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $log -Append -Encoding utf8
