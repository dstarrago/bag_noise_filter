/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Utils;

import java.util.Random;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.PropositionalToMultiInstance;

/**
 *
 * @author Danel
 */
public class StablePropositionalToMultiInstance extends PropositionalToMultiInstance {
  
  private static final long serialVersionUID = -671923623143594252L;
  
  /**
   * Signify that this batch of input to the filter is finished. 
   * If the filter requires all instances prior to filtering,
   * output() may now be called to retrieve the filtered instances.
   *
   * @return true if there are instances pending output
   * @throws IllegalStateException if no input structure has been defined
   */
  @Override
  public boolean batchFinished() {

    if (getInputFormat() == null) {
      throw new IllegalStateException("No input instance format defined");
    }

    Instances input = getInputFormat();
    //input.sort(0);   // make sure that bagID is sorted
    Instances output = getOutputFormat();
    Instances bagInsts = output.attribute(1).relation();
    Instance inst = new DenseInstance(bagInsts.numAttributes());
    inst.setDataset(bagInsts);

    double bagIndex   = input.instance(0).value(0);
    double classValue = input.instance(0).classValue(); 
    double bagWeight  = 0.0;

    // Convert pending input instances
    for(int i = 0; i < input.numInstances(); i++) {
      double currentBagIndex = input.instance(i).value(0);

      // copy the propositional instance value, except the bagIndex and the class value
      for (int j = 0; j < input.numAttributes() - 2; j++) 
        inst.setValue(j, input.instance(i).value(j + 1));
      inst.setWeight(input.instance(i).weight());

      if (currentBagIndex == bagIndex){
        bagInsts.add(inst);
        bagWeight += inst.weight();
      }
      else{
        addBag(input, output, bagInsts, (int) bagIndex, classValue, bagWeight);

        bagInsts   = bagInsts.stringFreeStructure();  
        bagInsts.add(inst);
        bagIndex   = currentBagIndex;
        classValue = input.instance(i).classValue();
        bagWeight  = inst.weight();
      }
    }

    // reach the last instance, create and add the last bag
    addBag(input, output, bagInsts, (int) bagIndex, classValue, bagWeight);

    if (getRandomize())
      output.randomize(new Random(getSeed()));
    
    for (int i = 0; i < output.numInstances(); i++)
      push(output.instance(i));
    
    // Free memory
    flushInput();

    m_NewBatch = true;
    m_FirstBatchDone = true;
    
    return (numPendingOutput() != 0);
  }
    
}
