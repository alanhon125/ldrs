from app.upload_doc.forms import UploadFileForm, FileFieldForm
from config import DOC_DIR
from django.views.decorators.csrf import csrf_exempt
from django.views.generic.edit import FormView
from django.http import HttpResponseRedirect
from django.shortcuts import render
 
# Imaginary function to handle an uploaded file.
def handle_uploaded_file(f):
    import os
    import re
    key1 = ['TS','term sheet']
    key2 = ['FA','facility agreement','facilities agreement']
    if re.match(r'.*'+r'.*|.*'.join(key1),f.name,flags=re.IGNORECASE):
        subfolder = 'TS/'
    elif re.match(r'.*'+r'.*|.*'.join(key2),f.name,flags=re.IGNORECASE):
        subfolder = 'FA/'
    else:
        subfolder = ''
    if not os.path.exists(f"{DOC_DIR}{subfolder}"):
        os.makedirs(f"{DOC_DIR}{subfolder}")
    with open(f"{DOC_DIR}{subfolder}{f.name}", "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)

@csrf_exempt
def upload_file(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES["file"])
            return HttpResponseRedirect("/upload_doc/upload_file")
    else:
        form = UploadFileForm()
    return render(request, "upload.html", {"form": form})

class FileFieldFormView(FormView):
    form_class = FileFieldForm
    template_name = "upload.html"  # Replace with your template.
    success_url = "/upload_doc/upload_file"  # Replace with your URL or reverse().

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        if form.is_valid():
            return self.form_valid(form)
        else:
            return self.form_invalid(form)

    def form_valid(self, form):
        files = form.cleaned_data["file_field"]
        for f in files:
            ...  # Do something with each file.
        return super().form_valid()