$ErrorActionPreference = 'Stop'
Set-Location 'D:\AI'
$py = "C:\Users\共产主义接班人\AppData\Local\Programs\Python\Python311\python.exe"
$cells = @(
  'gpt2:backdoor:narrative_context:20260718',
  'gpt2:backdoor:syntactic_clause:20260717',
  'gpt2:backdoor:syntactic_clause:20260718'
)
foreach ($cell in $cells) {
  $tag = $cell -replace ':','_'
  $log = "D:\AI\runs\train_$tag.log"
  $err = "D:\AI\runs\train_$tag.err"
  "$(Get-Date -Format o) START $cell" | Out-File -Append 'D:\AI\runs\dev_backdoor_batch.log' -Encoding utf8
  & $py -m scripts.run_implicit_matrix --matrix configs/implicit_benchmark_matrix.yaml --execute --cell $cell *>&1 | Tee-Object -FilePath $log
  $rc = $LASTEXITCODE
  "$(Get-Date -Format o) END $cell rc=$rc" | Out-File -Append 'D:\AI\runs\dev_backdoor_batch.log' -Encoding utf8
  if ($rc -ne 0) { "CELL FAILED: $cell rc=$rc" | Out-File -Append 'D:\AI\runs\dev_backdoor_batch.log' -Encoding utf8; break }
}
"$(Get-Date -Format o) BATCH DONE" | Out-File -Append 'D:\AI\runs\dev_backdoor_batch.log' -Encoding utf8
