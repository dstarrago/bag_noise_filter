/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MIFilters.instance;

import SIFilters.KeelNoiseFilter;
import Utils.AveHausdorffDistance;
import weka.core.Instances;
import weka.core.Instance;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.DenseInstance;
import weka.core.Utils;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.NonSparseToSparse;

/**
 * Nearest neighbor Instance Noise Filter is a multi-instance noise filter 
 * that works at the instance level.
 * 
 * @author Danel
 */
public class NNInstanceNoiseFilter extends AbstractInstanceNoiseFilter {
  
  public static final int VOTE_SCHEME_MINIMUM = 1;
  public static final int VOTE_SCHEME_MAJORITY = 2;
  public static final int VOTE_SCHEME_UNANIMITY = 3;
//  public static final int DEFAULT_VOTE_SCHEME = VOTE_SCHEME_UNANIMITY;
  public static final int DEFAULT_VOTE_SCHEME = VOTE_SCHEME_MAJORITY;
  public static final int DEFAULT_NEIGHBORS_NUMBER = 5; // 5
  public static final int SPARSE_DATASET_THRESHOLD = 250;
  private int voteScheme;
  private int numNeighbors; 
  private int voteThreshold;

  private ArrayList<Instance> classA;
  private ArrayList<Instance> classB;
  private ArrayList<Integer> classABagIndexes;
  private ArrayList<Integer> classBBagIndexes;
  private int numNeighborsToUseWithA;
  private int numNeighborsToUseWithB;
  private double[][] interBagDistances;
  private boolean[] decisions;
  private int numFiltered;
  private int[] instanceIndexes;
  private Instances siDataset;
  
  public NNInstanceNoiseFilter(Instances dataset, 
          Class<? extends KeelNoiseFilter> siFilterClass, int voteScheme, int numNeighbors) {
    super(dataset, siFilterClass);
    this.voteScheme = voteScheme;
    this.numNeighbors = numNeighbors;
    updateVoteThreshold();
  }

  public NNInstanceNoiseFilter(Instances dataset, Class<? extends KeelNoiseFilter> siFilterClass) {
    this(dataset, siFilterClass, DEFAULT_VOTE_SCHEME, DEFAULT_NEIGHBORS_NUMBER);
  }
  
  private void updateVoteThreshold() {
    switch (voteScheme) {
      case VOTE_SCHEME_MINIMUM: voteThreshold = 1; break;
      case VOTE_SCHEME_MAJORITY: voteThreshold = numNeighbors / 2 + 1; break;
      case VOTE_SCHEME_UNANIMITY: voteThreshold = numNeighbors; break;
    }
  }

  @Override
  public void apply() {
    initialize();
    organizeBagsPerClass();
    computeInterBagDistances();
    filterA();
    filterB();
    setDecisions(decisions);
    setNumFiltered(numFiltered);
    setNewMIDataset(getFilteredDataset());
  }
  
  private void initialize() {
    instanceIndexes = new int[getMIDataset().numInstances()];
    int instanceCount = 0;
    for (int i = 0; i < getMIDataset().numInstances(); i++) {
      instanceIndexes[i] = instanceCount;
      instanceCount += getMIDataset().get(i).relationalValue(1).numInstances();
    }
    decisions = new boolean[instanceCount];
  }
  
  private void organizeBagsPerClass() {
    classA = new ArrayList<>();
    classB = new ArrayList<>();
    classABagIndexes = new ArrayList<>();
    classBBagIndexes = new ArrayList<>();
    for (int i = 0; i < getMIDataset().numInstances(); i++) {
      Instance bag = getMIDataset().get(i);
      if (bag.classValue() == 0) {
        classA.add(bag);
        classABagIndexes.add(i);
      } else {
        classB.add(bag);
        classBBagIndexes.add(i);
      }
    }
    numNeighborsToUseWithA = (numNeighbors < classB.size())? numNeighbors: classB.size();
    numNeighborsToUseWithB = (numNeighbors < classA.size())? numNeighbors: classA.size();
  }
  
  private void computeInterBagDistances() {
    interBagDistances = new double[classA.size()][classB.size()];
    if (getMIDataset().attribute(1).relation().numAttributes() >= SPARSE_DATASET_THRESHOLD) {
      try {
        Instances compactData = toSparseBag(getMIDataset());
        for (int i = 0; i < classA.size(); i++) {
          for (int j = 0; j < classB.size(); j++) {
            int Ai = classABagIndexes.get(i);
            int Bj = classBBagIndexes.get(j);
            interBagDistances[i][j] = AveHausdorffDistance.sparseEuclideanMeasure(compactData.get(Ai), compactData.get(Bj));
          }
        }
        } catch (Exception ex) {
        Logger.getLogger(NNInstanceNoiseFilter.class.getName()).log(Level.SEVERE, null, ex);
      }
    } else {
      for (int i = 0; i < classA.size(); i++) {
        for (int j = 0; j < classB.size(); j++) {
          interBagDistances[i][j] = AveHausdorffDistance.denseEuclideanMeasure(classA.get(i), classB.get(j));
        }
      }
    }
  }
  
  public Instances toSparseBag(Instances miDataset) throws Exception {
    NonSparseToSparse filter = new NonSparseToSparse();
    Instances compactDataset = miDataset.stringFreeStructure();
    for (int i = 0; i < miDataset.numInstances(); i++) {
      Instances bag = miDataset.get(i).relationalValue(1);
      Instances newBag = new Instances(bag);
      filter.setInputFormat(newBag);   
      Instances compBag = Filter.useFilter(newBag, filter);
      int bagIndex = compactDataset.attribute(1).addRelation(compBag);
      double[] attrVals = new double[3];
      attrVals[0] = miDataset.get(i).value(0);
      attrVals[1] = bagIndex;
      attrVals[2] = miDataset.get(i).value(2);
      Instance newMIExemplar = new DenseInstance(1, attrVals);
      compactDataset.add(newMIExemplar);
    }
    return compactDataset;
  }
  
  private void filterA() {
    for (int i = 0; i < classA.size(); i++) {
      Instance X = classA.get(i);
      Instances siX = toSimpleInstance(X);
      int[] noisyInstanceVotes = new int[siX.size()];
      ArrayList<Instance> nearestNeighbors = getANN(i);
      for (int j = 0; j < nearestNeighbors.size(); j++) {
        Instance Z = nearestNeighbors.get(j);
        Instances siZ = toSimpleInstance(Z);
        Instances DS = merge(siX, siZ);
        KeelNoiseFilter siFilter = getSingleInstanceFilter(DS);
        for (int k = 0; k < siX.size(); k++) {
          if (siFilter.isNoisy(k)) {
            noisyInstanceVotes[k]++;
          }
        }
      }
      for (int k = 0; k < siX.size(); k++) {
        if (noisyInstanceVotes[k] >= voteThreshold) {
          int mappedIndex = instanceIndexes[classABagIndexes.get(i)] + k;
          decisions[mappedIndex] = true;
          numFiltered++;
        }
      }
    }
  }
  
  private void filterB() {
    for (int i = 0; i < classB.size(); i++) {
      Instance X = classB.get(i);
      Instances siX = toSimpleInstance(X);
      int[] noisyInstanceVotes = new int[siX.size()];
      ArrayList<Instance> nearestNeighbors = getBNN(i);
      for (int j = 0; j < nearestNeighbors.size(); j++) {
        Instance Z = nearestNeighbors.get(j);
        Instances siZ = toSimpleInstance(Z);
        Instances DS = merge(siX, siZ);
        KeelNoiseFilter siFilter = getSingleInstanceFilter(DS);
        for (int k = 0; k < siX.size(); k++) {
          if (siFilter.isNoisy(k)) {
            noisyInstanceVotes[k]++;
          }
        }
      }
      for (int k = 0; k < siX.size(); k++) {
        if (noisyInstanceVotes[k] >= voteThreshold) {
          int mappedIndex = instanceIndexes[classBBagIndexes.get(i)] + k;
          decisions[mappedIndex] = true;
          numFiltered++;
        }
      }
    }
  }
  
  private Instances toSimpleInstance(Instance bag) {
    Instances internalDS = bag.relationalValue(1);
    Instances siData = new Instances(internalDS, 0);
    for (int i = 0; i < internalDS.numInstances(); i++) {
      siData.add((Instance)internalDS.get(i).copy());
    }
    Add addAttr = new Add();
    try {
      addAttr.setOptions(new String[] {"-T","NOM", "-N", "class", "-L", "A,B", "-C", "last"});
      addAttr.setInputFormat(siData);
      siData = Filter.useFilter(siData, addAttr); // Add class attribute
    } catch (Exception ex) {
      Logger.getLogger(NNInstanceNoiseFilter.class.getName()).log(Level.SEVERE, null, ex);
    }
    siData.setClassIndex(siData.numAttributes() - 1);
    for (int i = 0; i < siData.numInstances(); i++) {
      siData.get(i).setClassValue(bag.classValue());
    }
    return siData;
  }
  
  private Instances merge(Instances A, Instances B) {
    Instances C = new Instances(A, A.size() + B.size());
    for (int i = 0; i < A.size(); i++) {
      C.add((Instance)A.get(i).copy());
    }
    for (int i = 0; i < B.size(); i++) {
      C.add((Instance)B.get(i).copy());
    }
    return C;
  }
  
  private ArrayList<Instance> getANN(int AIndex) {
    double[] dist  = java.util.Arrays.copyOf(interBagDistances[AIndex], classB.size());
    int[] sortedBIndex = Utils.sort(dist);  // Sort in ascending order
    ArrayList<Instance> nearestNeighbors = new ArrayList<>(numNeighborsToUseWithA);
    for (int i = 0; i < numNeighborsToUseWithA; i++) {
      nearestNeighbors.add(classB.get(sortedBIndex[i]));
    }
    return nearestNeighbors;
  }
  
  private ArrayList<Instance> getBNN(int BIndex) {
    double[] dist = new double[classA.size()]; 
    for (int i = 0; i < classA.size(); i++) {
      dist[i] = interBagDistances[i][BIndex];
    }
    int[] sortedAIndex = Utils.sort(dist);  // Sort in ascending order
    ArrayList<Instance> nearestNeighbors = new ArrayList<>(numNeighborsToUseWithB);
    for (int i = 0; i < numNeighborsToUseWithB; i++) {
      nearestNeighbors.add(classA.get(sortedAIndex[i]));
    }
    return nearestNeighbors;
  }
  
  public void setVoteScheme(int scheme) {
    voteScheme = scheme;
    updateVoteThreshold();
  }
  
  public void setNeighborsNumber(int val) {
    numNeighbors = val;
    updateVoteThreshold();
  }
  
  private Instances getFilteredDataset() {
    try {
      siDataset = toSimpleInstance(getMIDataset());
      Instances filteredSIData = new Instances(siDataset, 0);
      for (int i = 0; i < siDataset.numInstances(); i++) {
        if (!decisions[i]) {
          filteredSIData.add(siDataset.get(i));
        }
      }
      Instances miData = toMultiInstance(filteredSIData);
      return miData;
    } catch (Exception ex) {
      Logger.getLogger(NNInstanceNoiseFilter.class.getName()).log(Level.SEVERE, null, ex);
      return null;
    }
  }
  
}
