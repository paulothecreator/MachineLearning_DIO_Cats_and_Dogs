# MachineLearning_DIO_Cats_and_Dogs
Transfer Learning em uma rede de Deep Learning na linguagem Python no ambiente COLAB.  
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyNBpTXpNaZ6gVCIh028n51+",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/paulothecreator/Diagrama-ER-da-Oficina/blob/main/Deeplearning.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!Wget https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "v1ZkIWYPk-io"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!unzip cats_and_dogs_filtered.zip"
      ],
      "metadata": {
        "id": "7MLtH4bClYyb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "id": "g0REyCzPgw58"
      },
      "outputs": [],
      "source": [
        "!rm -rf cats_and_dogs_filtered.zip"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install tensorflow"
      ],
      "metadata": {
        "id": "EvUHvThDk-yf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import matplotlib.pyplot as plt\n",
        "import tensorflow as tf\n"
      ],
      "metadata": {
        "id": "LRU2Os0fk-17"
      },
      "execution_count": 45,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "dataset_dir = os.path.join(os.getcwd(),'cats_and_dogs_filtered')\n",
        "\n",
        "dataset_train_dir = os.path.join(dataset_dir, 'train')\n",
        "dataset_train_cats_len = len(os.listdir(os.path.join(dataset_train_dir, 'cats')))\n",
        "dataset_train_dogs_len = len(os.listdir(os.path.join(dataset_train_dir, 'dogs')))\n",
        "\n",
        "dataset_validation_dir = os.path.join(dataset_dir, 'validation')\n",
        "dataset_validation_cats_len = len(os.listdir(os.path.join(dataset_validation_dir, 'cats')))\n",
        "dataset_validation_dogs_len = len(os.listdir(os.path.join(dataset_validation_dir, 'dogs')))\n",
        "\n",
        "print('train Cats: %s' % dataset_train_cats_len)\n",
        "print('train Dogs: %s' % dataset_train_dogs_len)\n",
        "print('Validation Cats: %s' % dataset_validation_cats_len)\n",
        "print('Validation Cats: %s' % dataset_validation_cats_len)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BB3Kbdb1k-5M",
        "outputId": "b7b2ce66-e734-4969-8497-b77ce0114fb8"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "train Cats: 1000\n",
            "train Dogs: 1000\n",
            "Validation Cats: 500\n",
            "Validation Cats: 500\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "image_width = 160\n",
        "image_height = 160\n",
        "image_color_channel = 3\n",
        "image_color_channel_size = 255\n",
        "image_size = (image_width, image_height)\n",
        "image_shape = image_size + (image_color_channel,)\n",
        "\n",
        "batch_size = 32 \n",
        "epochs = 20\n",
        "learning_rate = 0.0001\n",
        "\n",
        "class_names = ('cat', 'dog')"
      ],
      "metadata": {
        "id": "_c54yuYZk-9l"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "dataset_train = tf.keras.preprocessing.image_dataset_from_directory(\n",
        "    dataset_train_dir,\n",
        "    image_size = image_size,\n",
        "    batch_size = batch_size,\n",
        "    shuffle = True\n",
        ")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "T3qiUknOk_BS",
        "outputId": "c3c4f8ae-bc51-479f-950e-db302966f613"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 2000 files belonging to 2 classes.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "dataset_validation = tf.keras.preprocessing.image_dataset_from_directory(\n",
        "    dataset_validation_dir,\n",
        "    image_size = image_size,\n",
        "    batch_size = batch_size,\n",
        "    shuffle = True\n",
        ")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RInLgXwltz5F",
        "outputId": "409c40d2-db11-4f1d-a091-8fa0d45aafcd"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 1000 files belonging to 2 classes.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "dataset_validation_cardinality = tf.data.experimental.cardinality(dataset_validation)\n",
        "dataset_validation_batches = dataset_validation_cardinality // 5\n",
        "\n",
        "dataset_test = dataset_validation.take(dataset_validation_batches)\n",
        "dataset_validation = dataset_validation.skip(dataset_validation_batches)\n",
        "\n",
        "print('Validation Dataset Cardinality: %d' % tf.data.experimental.cardinality(dataset_validation))\n",
        "print('Test Dataset Cardinality: %d' % tf.data.experimental.cardinality(dataset_test))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VoZH68yzt0GK",
        "outputId": "4be829b1-c4e7-4b96-ae0b-06aa4b08b53b"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Validation Dataset Cardinality: 26\n",
            "Test Dataset Cardinality: 6\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def plot_dataset(dataset):\n",
        "\n",
        "   plt.gcf(). clear()\n",
        "   plt.figure(figsize = (15, 15))\n",
        "\n",
        "   for features, labels in dataset. take(1):\n",
        "\n",
        "     for i in range(9):\n",
        "\n",
        "       plt.subplot(3, 3, i = 1)\n",
        "       plt.axis('off')\n",
        "\n",
        "       plt.inshow(features[i].numpy().astype('uint8'))\n",
        "       plt.title(class_names(labels[i]))"
      ],
      "metadata": {
        "id": "54FiOCozzQrf"
      },
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "plot_dataset(dataset_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 345
        },
        "id": "Y6leis7qzQ2J",
        "outputId": "91e54292-0aeb-4919-ab68-4ed5cd7843e1"
      },
      "execution_count": 37,
      "outputs": [
        {
          "output_type": "error",
          "ename": "ValueError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-37-2b46251ca113>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mplot_dataset\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdataset_train\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m;\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m<ipython-input-25-1dacce39ba3a>\u001b[0m in \u001b[0;36mplot_dataset\u001b[0;34m(dataset)\u001b[0m\n\u001b[1;32m      8\u001b[0m      \u001b[0;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m9\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      9\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 10\u001b[0;31m        \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msubplot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m3\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m3\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mi\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     11\u001b[0m        \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0maxis\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'off'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     12\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/matplotlib/pyplot.py\u001b[0m in \u001b[0;36msubplot\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m   1028\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1029\u001b[0m     \u001b[0mfig\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgcf\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1030\u001b[0;31m     \u001b[0ma\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mfig\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0madd_subplot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1031\u001b[0m     \u001b[0mbbox\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0ma\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbbox\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1032\u001b[0m     \u001b[0mbyebye\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/matplotlib/figure.py\u001b[0m in \u001b[0;36madd_subplot\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1417\u001b[0m                     \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_axstack\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mremove\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0max\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1418\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1419\u001b[0;31m             \u001b[0ma\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0msubplot_class_factory\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mprojection_class\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1420\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1421\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_add_axes_internal\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0ma\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/matplotlib/axes/_subplots.py\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, fig, *args, **kwargs)\u001b[0m\n\u001b[1;32m     69\u001b[0m                 \u001b[0;31m# num - 1 for converting from MATLAB to python indexing\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     70\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 71\u001b[0;31m             \u001b[0;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mf'Illegal argument(s) to subplot: {args}'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     72\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     73\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mupdate_params\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mValueError\u001b[0m: Illegal argument(s) to subplot: (3, 3)"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 0 Axes>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1080x1080 with 0 Axes>"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plot_dataset(dataset_validation)"
      ],
      "metadata": {
        "id": "6JPnxMlG4Wej"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "plot_dataset(dataset_test)"
      ],
      "metadata": {
        "id": "9ZbBBjOy4WqO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model = tf.keras.models.Sequential([\n",
        "        tf.keras.layers.experimental.preprocessing.Rescaling(\n",
        "        1. / image_color_channel_size,\n",
        "        input_shape = image_shape\n",
        "    ),\n",
        "    tf.keras.layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),\n",
        "    tf.keras.layers.MaxPooling2D(),\n",
        "    tf.keras.layers.Conv2D(32, 3, padding = 'same', activation = 'relu'),\n",
        "    tf.keras.layers.MaxPooling2D(),\n",
        "    tf.keras.layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),\n",
        "    tf.keras.layers.MaxPooling2D(),\n",
        "    tf.keras.layers.Flatten(),\n",
        "    tf.keras.layers.Dense(128, activation = 'relu'),\n",
        "    tf.keras.layers.Dense(1, activation = 'sigmoid')\n",
        "])\n",
        "model.compile(\n",
        "    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),\n",
        "    loss = tf.keras.losses.BinaryCrossentropy(),\n",
        "    metrics = ['accuracy']\n",
        ")\n",
        "model.summary()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-DksygBnt0Rk",
        "outputId": "b56f1256-d9cd-4ed5-83cd-3d4c3143d9a6"
      },
      "execution_count": 42,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"sequential_2\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " rescaling_4 (Rescaling)     (None, 160, 160, 3)       0         \n",
            "                                                                 \n",
            " conv2d_9 (Conv2D)           (None, 160, 160, 16)      448       \n",
            "                                                                 \n",
            " max_pooling2d_9 (MaxPooling  (None, 80, 80, 16)       0         \n",
            " 2D)                                                             \n",
            "                                                                 \n",
            " conv2d_10 (Conv2D)          (None, 80, 80, 32)        4640      \n",
            "                                                                 \n",
            " max_pooling2d_10 (MaxPoolin  (None, 40, 40, 32)       0         \n",
            " g2D)                                                            \n",
            "                                                                 \n",
            " conv2d_11 (Conv2D)          (None, 40, 40, 64)        18496     \n",
            "                                                                 \n",
            " max_pooling2d_11 (MaxPoolin  (None, 20, 20, 64)       0         \n",
            " g2D)                                                            \n",
            "                                                                 \n",
            " flatten_3 (Flatten)         (None, 25600)             0         \n",
            "                                                                 \n",
            " dense_6 (Dense)             (None, 128)               3276928   \n",
            "                                                                 \n",
            " dense_7 (Dense)             (None, 1)                 129       \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 3,300,641\n",
            "Trainable params: 3,300,641\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def plot_dataset_predications(dataset):\n",
        "\n",
        "  features, labels = dataset.as_numpy_iterator(), next()\n",
        "\n",
        "  predictions = model.predict_on_batch(features), flatten()\n",
        "  predictions = tf.where(predictions < 0.5, 0, 1)\n",
        "\n",
        "  print('labels:      %s' % labels)\n",
        "  print('Predictions: %s' % predictions.numpy())\n",
        "\n",
        "  plt.gcf().clear()\n",
        "  plt.figure(figsize = (15, 15))\n",
        "  for i in range(9):\n",
        "\n",
        "    plt.subplot(3, i + 1)\n",
        "    plt.axis('off')\n",
        "     \n",
        "    plt.imshow(features[i].astype('uint8'))\n",
        "    plt.title(class_names(predictions[i]))"
      ],
      "metadata": {
        "id": "4h5-Z2aR5Zwl"
      },
      "execution_count": 43,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "plot_dataset_predictions(dataset_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 165
        },
        "id": "uIGE3e-U5Z14",
        "outputId": "435bfafb-0c4c-4437-b1e4-32e41b143edb"
      },
      "execution_count": 46,
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-46-f8344839dbad>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mplot_dataset_predictions\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdataset_test\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m: name 'plot_dataset_predictions' is not defined"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "ofXS9qDRt0a1"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
