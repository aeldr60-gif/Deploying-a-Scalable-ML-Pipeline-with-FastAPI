# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a RandomForestClassifier, trained to predict whether individuals earn more or less than $50K per anum, based on US Census demographic data. It uses:
 - OneHotEncoder (categorical features)
 - LabelBinarizer (salary label)
 - RandomForestClassifier

The model and encoder are saved as .pkl files in the model folder.

## Intended Use

Intended uses of this model are:
- Providing salary-level predicitons

## Training Data

The training dataset is the census.csv file, which contains features such as:
- age
- workclass
- education
- marital-status
- occupation
- race
- etc.

## Evaluation Data

Evaluation data comes from the 20% split created in train_model.py. No external validation was used.

## Metrics

Precision: 0.7361 | Recall: 0.6314 | F1: 0.6797

## Ethical Considerations

Predictive models can potentially reinforce historical inequalities, especially on features such as race and sex.

They should not be used to make consequential decisions about individuals based on these types of features.

## Caveats and Recommendations

This model was only trained on census data and may not be generalizable to the populations not represented.
