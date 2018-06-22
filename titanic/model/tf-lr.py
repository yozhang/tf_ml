import tensorflow as tf
import numpy as np
import pandas as pd

_CSV_COLUMNS = [
    'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
]
_CSV_COLUMN_DEFAULTS=[[''],[1],[1],[''],[''],[0.0],[0],[0], [''],[0.0],[''],['']]

_CSV_COLUMN_DEFAULTS_TEST =[[''],[1],[''],[''],[0.0],[0],[0], [''],[0.0],[''],['']]

_CSV_COLUMNS_TEST = [
    'PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
]

TICKET_TYPE = ['A/5' 'PC' 'STON/O2.' 'AAAA' 'PP' 'A/5.' 'C.A.' 'A./5.' 'SC/Paris'
 'S.C./A.4.' 'A/4.' 'CA' 'S.P.' 'S.O.C.' 'SO/C' 'W./C.' 'SOTON/OQ'
 'W.E.P.' 'STON/O' 'A4.' 'C' 'SOTON/O.Q.' 'SC/PARIS' 'S.O.P.' 'A.5.' 'Fa'
 'CA.' 'LINE' 'F.C.C.' 'W/C' 'SW/PP' 'SCO/W' 'P/PP' 'SC' 'SC/AH' 'A/S'
 'A/4' 'WE/P' 'S.W./PP' 'S.O./P.P.' 'F.C.' 'SOTON/O2' 'S.C./PARIS'
 'C.A./SOTON']


def input_fn(data_file, num_epochs, shuffle, batch_size, is_train=True):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have either run data_download.py or '
      'set both arguments --train_data and --test_data.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, 
    record_defaults= _CSV_COLUMN_DEFAULTS if is_train else _CSV_COLUMN_DEFAULTS_TEST,
    na_value='' )
    features = dict(zip(_CSV_COLUMNS if is_train else _CSV_COLUMNS_TEST, columns))
    features.pop('PassengerId')
    labels = tf.constant(1, dtype=tf.int32)
    if is_train:
        labels = features.pop('Survived')
    return features, labels
    
  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

#  if shuffle:
#    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

  dataset = dataset.skip(1)
  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  if is_train:
      features, labels = iterator.get_next()
      return features, labels
  else:
      return iterator.get_next()

def build_model_columns():
    age = tf.feature_column.numeric_column('Age')
    fare = tf.feature_column.numeric_column('Fare',dtype=float)
    pclass = tf.feature_column.categorical_column_with_identity('Pclass', 6, default_value=5)
    sex = tf.feature_column.categorical_column_with_vocabulary_list('Sex', ['male', 'female'])
    sibsp = tf.feature_column.categorical_column_with_identity('SibSp', 11, default_value=10)
    parch = tf.feature_column.categorical_column_with_identity('Parch', 11, default_value=10)
    ticket = tf.feature_column.categorical_column_with_vocabulary_list('Ticket', TICKET_TYPE)
#     fare = tf.feature_column.categorical_column_with_hash_bucket('Fare', hash_bucket_size=1000)
    cabin = tf.feature_column.categorical_column_with_vocabulary_list('Cabin', ['Z', 'C', 'E', 'G', 'D', 'A', 'B', 'F', 'T'])
    embarded = tf.feature_column.categorical_column_with_vocabulary_list('Embarked', ['S', 'C', 'Q'])
    
    age_buckets = tf.feature_column.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    base_columns = [
        age_buckets, pclass, sibsp, parch, ticket, sex, fare, cabin, embarded
    ]
    crossed_columns = [
        tf.feature_column.crossed_column(
            ['SibSp', 'Parch'], hash_bucket_size=100
        ),
        tf.feature_column.crossed_column([age_buckets, 'Sex'], hash_bucket_size=100)
    ]
    wide_columns = base_columns + crossed_columns
    deep_columns = [
        age,
        tf.feature_column.indicator_column(pclass),
        tf.feature_column.indicator_column(sibsp),
        tf.feature_column.indicator_column(parch),
        tf.feature_column.indicator_column(cabin),
        tf.feature_column.indicator_column(ticket),
        tf.feature_column.indicator_column(embarded)
    ]
    return wide_columns, deep_columns

def build_estimator(model_dir, model_type):
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [100, 75, 50, 25]
    
    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU':0}))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            config=run_config)
    
def main(args):   
    model_dir = '/Users/zhangyong/Downloads/model_dir'
    train_path = './data/train_pro.csv'
    test_path = './data/test_pro.csv'
    model_type= ''
    model = build_estimator(model_dir, model_type)
    if args[0] is 'train':    
        for n in range(20):
            model.train(input_fn=lambda: input_fn(train_path, 2, True, 40, True))
        
        results = model.evaluate(input_fn=lambda: input_fn(test_path, 1, False, 40, False))

        print('Results at epoch', (n + 1) * 2)
        print('-' * 60)

        for key in sorted(results):
            print('%s: %s' % (key, results[key]))
    elif args[0] is 'predict':
        results = model.predict(input_fn=lambda: input_fn(test_path, 1, False, 40, False))
        vals = []
        for l in results:
            vals.append(l['class_ids'][0])
        tr = pd.read_csv(test_path)
        out = pd.DataFrame(tr, columns=['PassengerId'])
        out['Survived'] = vals
        out.to_csv('/Users/zhangyong/Downloads/tf_out.csv', index=False)
        

tf.app.run(main=main, argv=['predict'])
