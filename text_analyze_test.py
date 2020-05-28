from server import analyze
import torch
from server import get_all_projects

projects = {}
print(str(torch.cuda.is_available()) + 'text analyze test')
req = {
  "project": "new",
  "text": "The following is a transcript from The Guardian."
}


projects = get_all_projects()
ret = analyze(req)
print(ret)
