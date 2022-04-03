from wtforms import Form, FloatField, validators, DecimalField
from wtforms.validators import NumberRange


class InputForm(Form):
    Field1 = FloatField(validators=[validators.InputRequired()])
    Field2 = FloatField(validators=[validators.InputRequired()])
    Field3 = FloatField(validators=[validators.InputRequired()])
    Field4 = FloatField(validators=[validators.InputRequired()])
    Field5 = FloatField(validators=[validators.InputRequired()])
    Field6 = FloatField(validators=[validators.InputRequired()])
    Field7 = FloatField(validators=[validators.InputRequired()])
    Field8 = FloatField(validators=[validators.InputRequired()])
    Field9 = FloatField(validators=[validators.InputRequired()])
    Field10 = FloatField(validators=[validators.InputRequired()])
