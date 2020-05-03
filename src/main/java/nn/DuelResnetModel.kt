package nn

import org.deeplearning4j.nn.graph.ComputationGraph

class DuelResnetModel {
    companion object {
        fun getVersion1(blocks: Int, numPlanes: Int): ComputationGraph {
            val builder = ResidualNetworkBuilder()
            val input = "in"

            builder.addInputs(input)
            val initBlock = "init"
            val convOut = builder.addConvBatchNormBlock(initBlock, input, numPlanes, true)
            val towerOut = builder.addResidualTower(blocks, convOut)
            val policyOut = builder.addPolicyHead(towerOut)
            builder.addOutputs(policyOut)

            val model = ComputationGraph(builder.build())
            model.init()
            return model
        }

        fun getVersion2(blocks: Int, numPlanes: Int): ComputationGraph {
            val builder = ResidualNetworkBuilder(learningRate = 0.01)
            val input = "in"

            builder.addInputs(input)
            val initBlock = "init"
            val convOut = builder.addConvBatchNormBlock(initBlock, input, numPlanes, true)
            val towerOut = builder.addResidualTower(blocks, convOut)
            val fromPolicyOut = builder.addFromPolicyHead(towerOut)
            val toPolicyOut = builder.addToPolicyHead(towerOut)
            builder.addOutputs(fromPolicyOut, toPolicyOut)

            val model = ComputationGraph(builder.build())
            model.init()
            return model
        }
    }
}