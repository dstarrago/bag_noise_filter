/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MIFilters.instance;

import weka.core.Instances;
import SIFilters.KeelNoiseFilter;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instance;

/**
 * Multi-instance noise filter that works at the instance level.
 * 
 * It is a meta filter that runs iteratively the SimpleInstanceNoiseFilter
 * until a given threshold of noise detection is reached.
 * 
 * @author Danel
 */
public class IterativeSimpleInstanceNoiseFilter extends AbstractInstanceNoiseFilter {
  
  private final double LOWEST_APPRECIABLE_NOISE_PERCENT = 1;
  private Instances siDataset;
  private boolean[] decisions;
  private int numIterations;
  
  public IterativeSimpleInstanceNoiseFilter(Instances dataset, 
          Class<? extends KeelNoiseFilter> siFilterClass) {
    super(dataset, siFilterClass);
  }
  
  @Override
  public void apply() {
    try {
      siDataset = toSimpleInstance(getMIDataset());
      decisions = new boolean[siDataset.numInstances()];
      Instances targetData = siDataset;
      int partialNumFilteredInstances;
      int significantNoiseThreshold;
      int numFilteredInstances = 0;
      do {
        Instances sid = removeID(targetData);
        significantNoiseThreshold = (int)(sid.numInstances() * 
                LOWEST_APPRECIABLE_NOISE_PERCENT / 100.0);
        KeelNoiseFilter siFilter = getSingleInstanceFilter(sid);
        Instances newSIDataset = new Instances(targetData, 0);
        for (int i = 0; i < targetData.numInstances(); i++) {
          if (!siFilter.decisions()[i]) {
            newSIDataset.add(targetData.get(i));
          }
        }
        partialNumFilteredInstances = siFilter.numNoisyExamples();
        numFilteredInstances += partialNumFilteredInstances;
        updateDecisions(siFilter.decisions());
        targetData = newSIDataset;
        numIterations++;
      } while (partialNumFilteredInstances > significantNoiseThreshold);
      setNewMIDataset(toMultiInstance(targetData));
      setDecisions(decisions); 
      setNumFiltered(numFilteredInstances);
    } catch (Exception ex) {
      Logger.getLogger(IterativeSimpleInstanceNoiseFilter.class.getName()).log(Level.SEVERE, null, ex);
    }
    super.apply();
  }

  private void updateDecisions(boolean[] partialDecisions) {
    int partialNoiseCount = 0;
    for (int i = 0; i < partialDecisions.length; i++) {
      if (partialDecisions[i]) {
        int count = partialNoiseCount++; 
        int index = 0;
        while (count < i || decisions[index]) {
          if (!decisions[index]) {
            count++;
          }
          index++;
        }
        decisions[index] = true;
      }
    }
  }
  
  public Instance getInstance(int index) {
    return siDataset.get(index);
  }
  
  public int getNumIterations() {
    return numIterations;
  }
  
}
