from django import forms


class CustomerForm(forms.Form):
    age = forms.IntegerField(
        min_value=18, max_value=100,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    sex = forms.ChoiceField(
        choices=[("male", "Male"), ("female", "Female")],
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    bmi = forms.FloatField(
        min_value=10, max_value=60,
        label="BMI",
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.1"}),
    )
    children = forms.IntegerField(
        min_value=0, max_value=10,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    smoker = forms.ChoiceField(
        choices=[("no", "Non-smoker"), ("yes", "Smoker")],
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    region = forms.ChoiceField(
        choices=[
            ("northeast", "Northeast"),
            ("northwest", "Northwest"),
            ("southeast", "Southeast"),
            ("southwest", "Southwest"),
        ],
        widget=forms.Select(attrs={"class": "form-select"}),
    )
