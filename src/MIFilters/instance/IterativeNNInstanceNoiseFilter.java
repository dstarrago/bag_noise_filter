/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MIFilters.instance;

import SIFilters.KeelNoiseFilter;
import weka.core.Instances;

/**
 *
 * @author Danel
 */
public class IterativeNNInstanceNoiseFilter extends GenericIterativeInstanceNoiseFilter {
  
  public IterativeNNInstanceNoiseFilter(Instances dataset, 
          Class<? extends KeelNoiseFilter> siFilterClass) throws Exception {
    super(dataset, NNInstanceNoiseFilter.class, siFilterClass);
  }
  
}
