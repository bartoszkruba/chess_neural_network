package nn

import org.deeplearning4j.nn.graph.ComputationGraph

class DuelResnetModel {
    companion object {
        public fun get(blocks: Int, numPlanes: Int): ComputationGraph {
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
    }
}