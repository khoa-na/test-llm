$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location $PSScriptRoot\..
$py = ".\.venv\Scripts\python.exe"
$log = "eval\results\gemma3way_run.log"

function Set-Model($name) {
  $c = Get-Content .env -Raw
  $c = $c -replace '(?m)^MODEL_NAME=.*$', "MODEL_NAME=$name"
  Set-Content .env $c -NoNewline -Encoding utf8
}
function Deploy() { & $py -m modal deploy modal_app.py *>> $log }
function RunSet($s) { & $py eval\run_eval_judge.py --set $s *>> $log }

"=== START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $log -Encoding utf8

# 1) Gemma: ca CHAT + RAG_LIVE (model moi, chua co ket qua nao)
"`n##### MODEL = google/gemma-4-e4b-it #####" | Out-File $log -Append -Encoding utf8
Set-Model "google/gemma-4-e4b-it"; Deploy
"--- chat ---"     | Out-File $log -Append -Encoding utf8; RunSet "chat"
"--- rag_live ---" | Out-File $log -Append -Encoding utf8; RunSet "rag_live"

# 2) DeepSeek: chay lai RAG_LIVE tren bo case da-fix (apples-to-apples voi Gemma)
"`n##### MODEL = deepseek-ai/DeepSeek-R1-0528-Qwen3-8B (rag_live clean) #####" | Out-File $log -Append -Encoding utf8
Set-Model "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"; Deploy
"--- rag_live ---" | Out-File $log -Append -Encoding utf8; RunSet "rag_live"

# 3) Qwen: chay lai RAG_LIVE tren bo case da-fix
"`n##### MODEL = Qwen/Qwen3.5-9B (rag_live clean) #####" | Out-File $log -Append -Encoding utf8
Set-Model "Qwen/Qwen3.5-9B"; Deploy
"--- rag_live ---" | Out-File $log -Append -Encoding utf8; RunSet "rag_live"

"`n=== DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $log -Append -Encoding utf8
