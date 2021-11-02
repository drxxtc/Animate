from django import forms
from progmod.models import Parametrs
from django.utils.translation import ugettext_lazy as _

class ParamForm(forms.Form):
    class Meta:
        model=Parametrs
        fields=('speed_point', 'num_point')
