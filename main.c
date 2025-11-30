#include "NeuralC.h"

void exibir_neuronio(Network *network)
{
    printf("Topologia: %d", network->topologia[0]);
    for (size_t i = 1; i < network->quant_layers + 1; i++)
    {
        printf(":%d", network->topologia[i]);
    }
    printf("\n");

    for (int i = 0; i < network->quant_layers; i++)
    {
       
        printf("\n____________________Camada %d______________________\n\n", i + 1);
        

        for (int j = 0; j < network->layers[i]->quant_neuronios; j++)
        {
            printf("____________________________________________________\n");
            printf("____________________Neuronio %d______________________\n", j + 1);
            printf("_____________________________________________________\n");

            for (int w = 0; w < network->layers[i]->neuron[j]->quant_pesos; w++)
            {
                printf("Peso %d: %f\n", w + 1, network->layers[i]->neuron[j]->pesos[w]);
            }
            printf("Bias: %f\n", network->layers[i]->neuron[j]->bias);
        }
    }
}

int main()
{
    srand(time(NULL));
    Network *net = NULL;
    int topologia[] = {2, 3, 2, 1};

    net = create_network(3, topologia);

    exibir_neuronio(net);

    forward((float[]){1.0, 0.0}, net);

    printf("Saida: %f\n", net->layers[net->quant_layers - 1]->neuron[net->layers[net->quant_layers - 1]->quant_neuronios -1]->saida);

    return 0;
}