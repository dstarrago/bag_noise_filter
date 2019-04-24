/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MIFilters.instance;

import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instance;
import weka.core.Instances;
import SIFilters.EFNoiseFilter;
import SIFilters.IPFNoiseFilter;
import SIFilters.RNGNoiseFilter;

/**
 *
 * @author Danel
 */
public class IterativeNNInstanceNoiseMultiFilter extends AbstractInstanceNoiseFilter {
  
  private final double LOWEST_APPRECIABLE_NOISE_PERCENT = 1;
  private int numIterations;
  private boolean[] decisions;
  private Instances siDataset;

  public IterativeNNInstanceNoiseMultiFilter(Instances dataset) throws Exception {
    super(dataset, null);
  }
  
  @Override
  public void apply() {
    try {
      siDataset = toSimpleInstance(getMIDataset());
    } catch (Exception ex) {
      Logger.getLogger(GenericIterativeInstanceNoiseFilter.class.getName()).log(Level.SEVERE, null, ex);
    }
    decisions = new boolean[siDataset.numInstances()];
    Instances targetMIData = getMIDataset();
    int partialNumFilteredInstances = 0;
    int significantNoiseThreshold;
    int numFilteredInstances = 0;
    do {
      significantNoiseThreshold = (int)(targetMIData.numInstances() * 
              LOWEST_APPRECIABLE_NOISE_PERCENT / 100.0);
      try {
        NNInstanceNoiseMultiFilter filter = new NNInstanceNoiseMultiFilter(targetMIData, 
        EFNoiseFilter.class, IPFNoiseFilter.class, RNGNoiseFilter.class);
        filter.apply();
        targetMIData = filter.getNewMIDataset();
        updateDecisions(filter.decisions());
        partialNumFilteredInstances = filter.numFiltered();
        numFilteredInstances += partialNumFilteredInstances;
      } catch (Exception ex) {
        Logger.getLogger(GenericIterativeInstanceNoiseFilter.class.getName()).log(Level.SEVERE, null, ex);
      }
      } while (partialNumFilteredInstances > significantNoiseThreshold);
    setNewMIDataset(targetMIData);
    setDecisions(decisions);
    setNumFiltered(numFilteredInstances);
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
