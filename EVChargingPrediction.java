import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.csv.*;
import java.nio.file.Files;
import java.nio.file.Paths;

public class EVChargingPrediction {
    public static void main(String[] args) throws IOException {
        // Load datasets
        List<double[]> data = new ArrayList<>();
        List<Double> labels = new ArrayList<>();
        
        try (var reader = Files.newBufferedReader(Paths.get("datasets/EV_charging_reports.csv"));
             var csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader())) {
            for (CSVRecord record : csvParser) {
                double plugDuration = Double.parseDouble(record.get("Duration_hours").replace(',', '.'));
                double isPrivate = record.get("Location").equals("Private") ? 1.0 : 0.0;
                double month = Double.parseDouble(record.get("Month"));
                double dayOfWeek = Double.parseDouble(record.get("Day_of_Week"));
                double trafficDensity = Double.parseDouble(record.get("Traffic_Density"));
                double chargingLoad = Double.parseDouble(record.get("El_kWh").replace(',', '.'));
                
                data.add(new double[]{plugDuration, isPrivate, month, dayOfWeek, trafficDensity});
                labels.add(chargingLoad);
            }
        }

        int trainSize = (int) (0.8 * data.size());
        DataSet allData = new DataSet(Nd4j.create(data.toArray(new double[0][0])), Nd4j.create(labels.stream().mapToDouble(d -> d).toArray(), new int[]{labels.size(), 1}));
        allData.shuffle();

        DataSet trainData = new DataSet(allData.getFeatures().getRows(0, trainSize), allData.getLabels().getRows(0, trainSize));
        DataSet testData = new DataSet(allData.getFeatures().getRows(trainSize, data.size()), allData.getLabels().getRows(trainSize, data.size()));
        
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(trainData);
        normalizer.transform(trainData);
        normalizer.transform(testData);

        // Define neural network configuration
        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(42)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(5).nOut(56)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(56).nOut(26)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(26).nOut(1).build())
                .build());
        
        model.init();
        
        // Train the model
        int epochs = 3000;
        for (int epoch = 0; epoch < epochs; epoch++) {
            model.fit(trainData);
            if ((epoch + 1) % 500 == 0) {
                double loss = model.score();
                System.out.println("Epoch " + (epoch + 1) + ", Loss: " + loss);
            }
        }

        // Evaluate model
        INDArray predictions = model.output(testData.getFeatures());
        double testLoss = testData.getLabels().sub(predictions).mul(testData.getLabels().sub(predictions)).meanNumber().doubleValue();
        System.out.println("Test Loss: " + testLoss);

        // Save model
        File modelFile = new File("models/model.zip");
        model.save(modelFile, true);
    }
}
