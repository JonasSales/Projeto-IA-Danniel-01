public class  Treinamento {


    public Treinamento() {

    }


    public void treinar(NeuralNetwork rede, double[][] entradas, double[][] saidas, int epocas){
        for (int epoca = 1; epoca <= epocas; epoca++) {
            for (int j = 0; j < entradas.length; j++) {
                rede.train(entradas[j], saidas[j]);
            }

            if (epoca % 25 == 0) {
                int acertos = 0;
                for (int j = 0; j < entradas.length; j++) {
                    double[] saidaPredita = rede.feedforward(entradas[j]);
                    boolean correto = true;
                    for (int k = 0; k < saidas[j].length; k++) {
                        int bitPredito = (saidaPredita[k] >= 0.5 ? 1 : 0);
                        if (bitPredito != (int) saidas[j][k]) {
                            correto = false;
                            break;
                        }
                    }
                    if (correto) acertos++;
                }

                double precisao = (acertos / (double) entradas.length) * 100.0;
                System.out.printf("Época %d - Precisão: %.2f%%%n", epoca, precisao);
            }
        }
    }

    public void resultados(NeuralNetwork rede,double[][] entradas){
        System.out.println("\nResultado Final:");
        for (double[] entrada : entradas) {
            double[] resultado = rede.feedforward(entrada);
            System.out.printf("Entrada: %.2f => Saída: ", entrada[0]);
            for (double v : resultado)
                System.out.print((v >= 0.5 ? 1 : 0) + " ");
            System.out.println();
        }
    }
}
