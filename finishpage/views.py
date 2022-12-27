from django.http import HttpResponse


def index(request):
    return HttpResponse("감사합니다")