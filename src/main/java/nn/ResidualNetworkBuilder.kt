package nn

import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.InputPreProcessor
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ActivationLayer
import org.deeplearning4j.nn.conf.layers.BatchNormalization
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.learning.config.Sgd

class ResidualNetworkBuilder(
        private val kernel: IntArray = intArrayOf(3, 3),
        private val strides: IntArray = intArrayOf(1, 1),
        private val convolutionMode: ConvolutionMode = ConvolutionMode.Same,
        private val learningRate: Double = 0.001
) {
    private val conf = NeuralNetConfiguration.Builder()
            .updater(Adam(learningRate))
            .weightInit(WeightInit.LECUN_NORMAL)
            .graphBuilder().setInputTypes(InputType.convolutional(8, 8, 13))

    fun addInputs(name: String) {
        conf.addInputs(name)
    }

    fun addOutputs(vararg names: String) {
        conf.setOutputs(*names)
    }

    fun build() = conf.build()

    //    Building block for AGZ residual blocks.
    //    conv2d -> batch norm -> ReLU
    fun addConvBatchNormBlock(blockName: String, inName: String, nIn: Int, useActivation: Boolean): String {
        val convName = "conv_$blockName"
        val bnName = "batch_norm_$blockName"
        val actName = "relu_$blockName"
        conf.addLayer(convName, ConvolutionLayer.Builder()
                .kernelSize(*kernel)
                .stride(*strides)
                .convolutionMode(convolutionMode)
                .nIn(nIn).nOut(256)
                .build(), inName)

        conf.addLayer(bnName, BatchNormalization.Builder()
                .nOut(255)
                .build(), convName)

        return if (useActivation) {
            conf.addLayer(actName, ActivationLayer.Builder().activation(Activation.RELU).build(), bnName)
            actName
        } else bnName
    }

    // Residual block for AGZ. Takes two conv-bn-relu blocks
    // and adds them to the original input
    fun addResidualBlock(blockNumber: Int, inName: String): String {
        val firstBlock = "residual_1_$blockNumber"
        val secondBlock = "residual_2_$blockNumber"
        val mergeBlock = "add_$blockNumber"
        val actBlock = "relu_$blockNumber"

        val firstBnOut = addConvBatchNormBlock(firstBlock, inName, 256, true)
        val secondBnOut = addConvBatchNormBlock(secondBlock, firstBnOut, 256, false)
        conf.addVertex(mergeBlock, ElementWiseVertex(ElementWiseVertex.Op.Add), firstBnOut, secondBnOut)
        conf.addLayer(actBlock, ActivationLayer.Builder().activation(Activation.RELU).build(), mergeBlock)
        return actBlock
    }

    // Build a tower of residual blocks.
    fun addResidualTower(numBlocks: Int, inName: String): String {
        var name = inName
        for (i in 0 until numBlocks) {
            name = addResidualBlock(i, name)
        }
        return name
    }

    fun addPolicyHead(inName: String): String {
        val convName = "policy_head_conv_"
        val bnName = "policy_head_batch_norm_"
        val actName = "policy_head_relu_"
        val denseName = "policy_head_output_"

        conf.addLayer(convName, ConvolutionLayer.Builder()
                .kernelSize(*kernel)
                .stride(*strides)
                .convolutionMode(convolutionMode)
                // reducing convolutions to 2 planes - one for starting position, one for where to move
                .nOut(2).nIn(256)
                .build(), inName)
        conf.addLayer(bnName, BatchNormalization.Builder()
                .nOut(2)
                .build(), convName)
        conf.addLayer(actName, ActivationLayer.Builder()
                .activation(Activation.SOFTMAX)
                .build(), bnName)
        conf.addLayer(denseName, OutputLayer.Builder()
                // number inputs: two planes with size 8x8
                .nIn(2 * 8 * 8).nOut(63 * 64)
                .build(), actName)

        val preProcessorMap = hashMapOf<String, InputPreProcessor>(
                denseName to CnnToFeedForwardPreProcessor(8, 8, 2)
        )
        conf.inputPreProcessors = preProcessorMap
        return denseName
    }

    // This output decides from which square piece should be moved
    fun addFromPolicyHead(inName: String): String {
        val convName = "from_policy_head_conv_"
        val bnName = "from_policy_head_batch_norm_"
        val actName = "from_policy_head_relu_"
        val denseName = "from_policy_head_output_"

        conf.addLayer(convName, ConvolutionLayer.Builder()
                .kernelSize(*kernel)
                .stride(*strides)
                .convolutionMode(convolutionMode)
                // reducing convolutions to 2 planes - one for starting position, one for where to move
                .nOut(2).nIn(256)
                .build(), inName)
        conf.addLayer(bnName, BatchNormalization.Builder()
                .nOut(2)
                .build(), convName)
        conf.addLayer(actName, ActivationLayer.Builder()
                .activation(Activation.SOFTMAX)
                .build(), bnName)
        conf.addLayer(denseName, OutputLayer.Builder()
                // number inputs: two planes with size 8x8
                .nIn(2 * 8 * 8).nOut(64)
                .build(), actName)

        val preProcessorMap = hashMapOf<String, InputPreProcessor>(
                denseName to CnnToFeedForwardPreProcessor(8, 8, 2)
        )
        conf.inputPreProcessors = preProcessorMap
        return denseName
    }

    // This output decides from which to square piece should be moved
    fun addToPolicyHead(inName: String): String {
        val convName = "to_policy_head_conv_"
        val bnName = "to_policy_head_batch_norm_"
        val actName = "to_policy_head_relu_"
        val denseName = "to_policy_head_output_"

        conf.addLayer(convName, ConvolutionLayer.Builder()
                .kernelSize(*kernel)
                .stride(*strides)
                .convolutionMode(convolutionMode)
                // reducing convolutions to 2 planes - one for starting position, one for where to move
                .nOut(2).nIn(256)
                .build(), inName)
        conf.addLayer(bnName, BatchNormalization.Builder()
                .nOut(2)
                .build(), convName)
        conf.addLayer(actName, ActivationLayer.Builder()
                .activation(Activation.SOFTMAX)
                .build(), bnName)
        conf.addLayer(denseName, OutputLayer.Builder()
                // number inputs: two planes with size 8x8
                .nIn(2 * 8 * 8).nOut(64)
                .build(), actName)

        val preProcessorMap = hashMapOf<String, InputPreProcessor>(
                denseName to CnnToFeedForwardPreProcessor(8, 8, 2)
        )
        conf.inputPreProcessors = preProcessorMap
        return denseName
    }

    // todo add value head
}