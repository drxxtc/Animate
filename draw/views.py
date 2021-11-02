from django.shortcuts import render, redirect
from .models import Parametrs 
from mod.calc import anime


# Create your views here.
def main_view(request):  
    '''if request=='POST':
        speed_point=request.POST.get('speed')
        num_point = request.POST.get('count')
        chart=anime(speed_point,  num_point)
        #return render(request, 'main.html', {'chart':chart})
        return redirect('draw:main_view', {'chart': chart})'''
    chart=anime(speed_point=1,  num_point=2, choice=1)
    return render(request, 'main.html', {'chart': chart})