# TensorFlow Dataloop Model Adapters

This repository contains TensorFlow model adapters for integration with the Dataloop platform.

## Available Models

1. MoViNet - Mobile video classification network [documentation](adapters/movinet/README.md).

Full Model Management documentation [here](https://dataloop.ai/docs).
Developers Documentation is [here](https://developers.dataloop.ai/tutorials/model_management/).

## Installation

You can add these models to your Dataloop project in two ways:

### Option 1: Dataloop Marketplace (UI)

The easiest way to use these models is to install them directly from the [Dataloop Marketplace](https://dataloop.ai/platform/marketplace/):

1. Navigate to the [Dataloop platform](https://dataloop.ai/)
2. Go to the "Marketplace" section
3. Search for "MoViNet"
4. Click "Install" to add the model to your project

This will make the model immediately available in your Library for use in annotations, model management, and pipelines.

### Option 2: Clone Via SDK

You can clone the pretrained model to your project using Python code.

First get all the public models:

```python
import dtlpy as dl

filters = dl.Filters(resource=dl.FILTERS_RESOURCE_MODEL)
filters.add(field='scope', values='public')

dl.models.list(filters=filters).print()
```

Select the pretrained model you want to clone and... clone:

```python
import dtlpy as dl

public_model = dl.models.get(model_id='')
project = dl.projects.get(project_name='My Project')
model = project.models.clone(from_model=public_model,
                             model_name='my_pretrained_movinet',
                             project_id=project.id)
```

## Finetune

If you want to finetune the model, you'll need to connect your dataset and train:

```python
dataset = project.datasets.get('Capybaras')
train_filter = dl.Filters(field='dir', values='/train')
validation_filter = dl.Filters(field='dir', values='/validation')
custom_model = project.models.clone(from_model=public_model,
                                    model_name='finetuning_mode',
                                    dataset=dataset,
                                    project_id=project.id,
                                    train_filter=train_filter,
                                    validation_filter=validation_filter)
```

Now you have a new model connected to your dataset, and you can initiate a training execution.
More information [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#train)

## License

The models in this repository are distributed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). When using these model adapters, you must comply with the terms of this license.

## Contributions

Help us get better! We welcome any contribution and suggestion to this repo.   
Open an issue for bug/features requests.