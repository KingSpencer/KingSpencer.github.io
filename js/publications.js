// Publications data
const publications = [
  {
    title: "Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems",
    authors: "Shangbin Feng, Zifeng Wang, Palash Goyal, Yike Wang, Weijia Shi, Huang Xia, Hamid Palangi, Luke Zettlemoyer, Yulia Tsvetkov, Chen-Yu Lee, Tomas Pfister",
    venue: {
      fullName: "arXiv",
      year: 2025
    },
    paperLink: "https://arxiv.org/abs/2502.04510",
    image: "paper_images/heterogeneous_swarms.png",
    tags: ["llm"],
    description: "We propose Heterogeneous Swarms, an algorithm to design multi-LLM systems by jointly optimizing model roles (i.e., multi-LLM workflow / topology) and model weights. Heterogeneous Swarms discovers multi-LLM systems with heterogeneous model roles and substantial collaborative gains, and benefits from the diversity of language models."
  },
  {
    title: "Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence",
    authors: "Shangbin Feng, Zifeng Wang, Yike Wang, Sayna Ebrahimi, Hamid Palangi, Lesly Miculicich, Achin Kulshrestha, Nathalie Rauschmayr, Yejin Choi, Yulia Tsvetkov, Chen-Yu Lee, Tomas Pfister",
    venue: {
      fullName: "International Conference on Machine Learning",
      abbreviation: "ICML",
      year: 2025
    },
    paperLink: "https://arxiv.org/abs/2410.11163",
    codeLink: "https://github.com/google-research/google-research/tree/master/model_swarm",
    image: "paper_images/modelswarms.png",
    tags: ["llm"],
    description: "We propose Model Swarms, a collaborative search algorithm to adapt LLMs via swarm intelligence, the collective behavior guiding individual systems. Specifically, Model Swarms starts with a pool of LLM experts and a utility function. Guided by the best-found checkpoints across models, diverse LLM experts collaboratively move in the weight space and optimize a utility function representing model adaptation objectives."
  },
  {
    title: "Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation",
    authors: "Fan Yin, Zifeng Wang, I-Hung Hsu, Jun Yan, Ke Jiang, Yanfei Chen, Jindong Gu, Long T. Le, Kai-Wei Chang, Chen-Yu Lee, Hamid Palangi, Tomas Pfister",
    venue: {
      fullName: "Association for Computational Linguistics",
      abbreviation: "ACL",
      year: 2025
    },
    paperLink: "https://arxiv.org/abs/2503.07826",
    image: "paper_images/magnet.png",
    tags: ["llm"],
    description: "We propose Magnet, a principled framework for synthesizing high-quality training trajectories to enhance the function calling capability of large language model agents in multi-turn conversations with humans. The framework is based on automatic and iterative translations from a function signature path to a sequence of queries and executable function calls."
  },
  {
    title: "In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents",
    authors: "Zhen Tan, Jun Yan, I-Hung Hsu, Rujun Han, Zifeng Wang, Long T. Le, Yiwen Song, Yanfei Chen, Hamid Palangi, George Lee, Anand Iyer, Tianlong Chen, Huan Liu, Chen-Yu Lee, Tomas Pfister",
    venue: {
      fullName: "Association for Computational Linguistics",
      abbreviation: "ACL",
      year: 2025
    },
    paperLink: "https://arxiv.org/abs/2503.08026",
    image: "paper_images/rmm_memory.png",
    tags: ["llm"],
    description: "We propose Reflective Memory Management (RMM), a novel mechanism for long-term dialogue agents, integrating forward- and backward-looking reflections: (1) Prospective Reflection, which dynamically summarizes interactions across granularities-utterances, turns, and sessions-into a personalized memory bank for effective future retrieval, and (2) Retrospective Reflection, which iteratively refines the retrieval in an online reinforcement learning (RL) manner based on LLMs' cited evidence."
  },
  {
    title: "Speculative Knowledge Distillation: Bridging the Teacher-Student Gap Through Interleaved Sampling",
    authors: "Wenda Xu, Rujun Han, Zifeng Wang, Long T. Le, Dhruv Madeka, Lei Li, William Yang Wang, Rishabh Agarwal, Chen-Yu Lee, Tomas Pfister",
    venue: {
      fullName: "International Conference on Learning Representations",
      abbreviation: "ICLR",
      year: 2025
    },
    paperLink: "https://arxiv.org/abs/2410.11325",
    codeLink: "https://github.com/google-research/google-research/tree/master/speculative_kd",
    image: "paper_images/skd.png",
    tags: ["llm"],
    description: "Speculative Knowledge Distillation (SKD) is a novel approach that leverages cooperation between student and teacher models to generate high-quality training data on-the-fly while aligning with the student's inference-time distribution. In SKD, the student proposes tokens, and the teacher replaces poorly ranked ones based on its own distribution, transferring high-quality knowledge adaptively."
  },
  {
    title: "Reverse Thinking Makes LLMs Stronger Reasoners",
    authors: "Justin Chih-Yao Chen, Zifeng Wang, Hamid Palangi, Rujun Han, Sayna Ebrahimi, Long Le, Vincent Perot, Swaroop Mishra, Mohit Bansal, Chen-Yu Lee, Tomas Pfister",
    venue: {
      fullName: "The Nations of the Americas Chapter of the Association for Computational Linguistics",
      abbreviation: "NAACL",
      year: 2025
    },
    paperLink: "https://arxiv.org/abs/2411.19865",
    image: "paper_images/revThink.png",
    tags: ["llm"],
    description: "To enable LLMs to perform reverse thinking, we introduce Reverse-Enhanced Thinking (RevThink), a framework composed of data augmentation and learning objectives. RevThink outperforms a standard fine-tuning method trained on 10x more forward reasoning, it also exhibits strong generalization to out-of-distribution held-out datasets."
  },
  {
    title: "Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting",
    authors: "Zilong Wang, Zifeng Wang, Long Le, Huaixiu Steven Zheng, Swaroop Mishra, Vincent Perot, Yuwei Zhang, Anush Mattapalli, Ankur Taly, Jingbo Shang, Chen-Yu Lee, Tomas Pfister",
    venue: {
      fullName: "International Conference on Learning Representations",
      abbreviation: "ICLR",
      year: 2025
    },
    paperLink: "https://arxiv.org/abs/2407.08223",
    image: "paper_images/speculativeRAG.png",
    tags: ["llm"],
    description: "Speculative RAG is a framework that leverages a larger generalist LM to efficiently verify multiple RAG drafts produced in parallel by a smaller, distilled specialist LM. (A complementary strategy to the CaLM paper below from a very different perspective!)",
    blogLink: "https://research.google/blog/speculative-rag-enhancing-retrieval-augmented-generation-through-drafting/"
  },
  {
    title: "TableRAG: Million-Token Table Understanding with Language Models",
    authors: "Si-An Chen, Lesly Miculicich, Julian Martin Eisenschlos, Zifeng Wang, Zilong Wang, Yanfei Chen, Yasuhisa Fujii, Hsuan-Tien Lin, Chen-Yu Lee, Tomas Pfister",
    venue: {
      fullName: "Neural Information Processing Systems",
      abbreviation: "NeurIPS",
      year: 2024
    },
    paperLink: "https://arxiv.org/abs/2410.04739",
    image: "paper_images/tableRAG.png",
    tags: ["llm"],
    description: "TableRAG is a Retrieval-Augmented Generation (RAG) framework specifically designed for LM-based table understanding. TableRAG leverages query expansion combined with schema and cell retrieval to pinpoint crucial information before providing it to the LMs.",
    codeLink: "https://github.com/google-research/google-research/tree/master/table_rag"
  },
  {
    title: "CaLM: Contrasting Large and Small Language Models to Verify Grounded Generation",
    authors: "I-Hung Hsu, Zifeng Wang, Long T Le, Lesly Miculicich, Nanyun Peng, Chen-Yu Lee, Tomas Pfister",
    venue: {
      fullName: "Findings of the Association for Computational Linguistics",
      abbreviation: "ACL",
      year: 2024
    },
    paperLink: "https://arxiv.org/abs/2406.05365",
    image: "paper_images/calm.png",
    tags: ["llm"],
    description: "CaLM is a novel verification framework that leverages the insight that a robust grounded response should be consistent with information derived solely from its cited sources. Our framework empowers smaller LMs, which rely less on parametric memory and excel at processing relevant information given a query, to validate the output of larger LMs."
  },
  {
    title: "Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization",
    authors: "Cheng-Yu Hsieh, Yung-Sung Chuang, Chun-Liang Li, Zifeng Wang, Long T. Le, Abhishek Kumar, James Glass, Alexander Ratner, Chen-Yu Lee, Ranjay Krishna, Tomas Pfister",
    venue: {
      fullName: "Findings of the Association for Computational Linguistics",
      abbreviation: "ACL",
      year: 2024
    },
    paperLink: "https://arxiv.org/abs/2406.16008",
    image: "paper_images/fitm.png",
    tags: ["llm"],
    description: "LLMs often struggle to capture relevant information in the middle of their input due to an intrinsic U-shaped attention bias, favoring tokens at the beginning and end. To address this, we propose a calibration mechanism called 'found-in-the-middle' to mitigate this bias, greatly improving context relevance and RAG performance."
  },
  {
    title: "CodecLM: Aligning Language Models with Tailored Synthetic Data",
    authors: "Zifeng Wang, Chun-Liang Li, Vincent Perot, Long T. Le, Jin Miao, Zizhao Zhang, Chen-Yu Lee, Tomas Pfister",
    venue: {
      fullName: "Findings of North American Chapter of the Association for Computational Linguistics",
      abbreviation: "NAACL",
      year: 2024
    },
    paperLink: "https://arxiv.org/abs/2404.05875",
    image: "paper_images/codeclm.png",
    tags: ["llm"],
    description: "CodecLM is a general framework for adaptively generating high-quality synthetic data for LLM alignment with different downstream instruction distributions and LLMs. Drawing on the Encode-Decode principles, we use LLMs as codecs to guide the data generation process.",
    blogLink: "https://research.google/blog/codeclm-aligning-language-models-with-tailored-synthetic-data/"
  },
  {
    title: "Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding",
    authors: "Zilong Wang, Hao Zhang, Chun-Liang Li, Julian Martin Eisenschlos, Vincent Perot, Zifeng Wang, Lesly Miculicich, Yasuhisa Fujii, Jingbo Shang, Chen-Yu Lee, Tomas Pfister",
    venue: {
      fullName: "International Conference on Learning Representations",
      abbreviation: "ICLR",
      year: 2024
    },
    paperLink: "https://arxiv.org/abs/2401.04398",
    image: "paper_images/chain-of-table.png",
    tags: ["llm"],
    description: "CHAIN-OF-TABLE enhances the reasoning capability of LLMs by leveraging tabular structures to express intermediate thoughts for table-based reasoning. It instructs LLMs to dynamically plan an operation chain according to the input table and its associated question."
  },
  {
    title: "QueryForm: A Simple Zero-shot Form Entity Query Framework",
    authors: "Zifeng Wang, Zizhao Zhang, Jacob Devlin, Chen-Yu Lee, Guolong Su, Hao Zhang, Jennifer Dy, Vincent Perot, Tomas Pfister",
    venue: {
      fullName: "Findings of the Association for Computational Linguistics",
      abbreviation: "ACL",
      year: 2023
    },
    paperLink: "https://arxiv.org/abs/2211.07730",
    image: "paper_images/queryform.png",
    tags: ["llm"],
    description: "QueryForm consists of a novel prompting-based framework for zero-shot document entity recognition with large language models (LLMs), and a large-scale weakly-supervised pre-training method on publicly available webpages."
  },
  {
    title: "DualHSIC: HSIC-Bottleneck and Alignment for Continual Learning",
    authors: "Zifeng Wang*, Zheng Zhan*, Yifan Gong, Yucai Shao, Stratis Ioannidis, Yanzhi Wang, Jennifer Dy",
    venue: {
      fullName: "International Conference on Machine Learning",
      abbreviation: "ICML",
      year: 2023
    },
    paperLink: "https://arxiv.org/abs/2305.00380",
    image: "paper_images/dualhsic.png",
    tags: ["cl"],
    description: "DualHSIC consists of two complementary components that stem from the Hilbert Schmidt independence criterion (HSIC): HSIC-Bottleneck for Rehearsal (HBR) lessens the inter-task interference and HSIC Alignment (HA) promotes task-invariant knowledge sharing.",
    codeLink: "https://github.com/zhanzheng8585/DualHSIC"
  },
  {
    title: "SparCL: Sparse Continual Learning on the Edge",
    authors: "Zifeng Wang*, Zheng Zhan*, Yifan Gong, Geng Yuan, Wei Niu, Tong Jian, Bin Ren, Stratis Ioannidis, Yanzhi Wang, Jennifer Dy",
    venue: {
      fullName: "Neural Information Processing Systems",
      abbreviation: "NeurIPS",
      year: 2022
    },
    paperLink: "https://arxiv.org/abs/2209.09476",
    image: "paper_images/sparcl.png",
    tags: ["cl"],
    description: "SparCL explores sparsity for efficient continual learning and achieves both training acceleration and accuracy preservation through the synergy of three aspects: weight sparsity, data efficiency, and gradient sparsity.",
    codeLink: "https://github.com/neu-spiral/SparCL"
  },
  {
    title: "DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning",
    authors: "Zifeng Wang, Zizhao Zhang, Sayna Ebrahimi, Ruoxi Sun, Han Zhang, Chen-Yu Lee, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, Tomas Pfister",
    venue: {
      fullName: "European Conference on Computer Vision",
      abbreviation: "ECCV",
      year: 2022
    },
    paperLink: "https://arxiv.org/abs/2204.04799",
    image: "paper_images/dualprompt.png",
    tags: ["cl"],
    description: "DualPrompt presents a novel approach to attach complementary prompts to the pre-trained backbone, and then formulates the continual learning objective as learning task-invariant and task-specific 'instructions'.",
    codeLink: "https://github.com/google-research/l2p"
  },
  {
    title: "Learning to Prompt for Continual Learning",
    authors: "Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, Tomas Pfister",
    venue: {
      fullName: "IEEE Conference on Computer Vision and Pattern Recognition",
      abbreviation: "CVPR",
      year: 2022
    },
    paperLink: "https://arxiv.org/abs/2112.08654",
    image: "paper_images/l2p.png",
    tags: ["cl"],
    description: "We propose a new learning paradigm for continual learning: our method learns to dynamically prompt (L2P) a pre-trained model to learn tasks sequentially under different task transitions.",
    codeLink: "https://github.com/google-research/l2p",
    blogLink: "https://ai.googleblog.com/2022/04/learning-to-prompt-for-continual.html"
  },
  {
    title: "Revisiting Hilbert-Schmidt Information Bottleneck for Adversarial Robustness",
    authors: "Zifeng Wang*, Tong Jian*, Aria Masoomi, Stratis Ioannidis, Jennifer Dy",
    venue: {
      fullName: "Neural Information Processing Systems",
      abbreviation: "NeurIPS",
      year: 2021
    },
    paperLink: "https://arxiv.org/abs/2106.02734",
    image: "paper_images/hbar.png",
    tags: ["misc"],
    description: "We investigate the HSIC (Hilbert-Schmidt independence criterion) bottleneck as a regularizer for learning an adversarially robust deep neural network classifier, both theoretically and empirically.",
    codeLink: "https://github.com/neu-spiral/HBaR"
  },
  {
    title: "Learn-Prune-Share for Lifelong Learning",
    authors: "Zifeng Wang*, Tong Jian*, Kaushik Chowdhury, Yanzhi Wang, Jennifer Dy, Stratis Ioannidis",
    venue: {
      fullName: "International Conference on Data Mining",
      abbreviation: "ICDM",
      year: 2020
    },
    paperLink: "https://arxiv.org/abs/2012.06956",
    image: "paper_images/lps.png",
    tags: ["cl"],
    description: "We propose a learn-prune-share (LPS) algorithm which addresses the challenges of catastrophic forgetting, parsimony, and knowledge reuse simultaneously."
  },
  {
    title: "Open-World Class Discovery with Kernel Networks",
    authors: "Zifeng Wang, Batool Salehi, Andrey Gritsenko, Kaushik Chowdhury, Stratis Ioannidis, Jennifer Dy",
    venue: {
      fullName: "International Conference on Data Mining",
      abbreviation: "ICDM",
      year: 2020
    },
    paperLink: "https://arxiv.org/abs/2012.06957",
    image: "paper_images/knet.png",
    tags: ["misc"],
    description: "We propose Class Discovery Kernel Network with Expansion (CD-KNet-Exp), a deep learning framework for open-world class discovery problem.",
    award: "Best Paper Candidate"
  }
];

// Function to create HTML for a publication
function createPublicationHTML(pub) {
  const carouselId = pub.image.split('/').pop().split('.')[0] + '-carousel';
  
  // Make Zifeng Wang bold
  const authorsWithBold = pub.authors.replace(/Zifeng Wang/g, '<strong>Zifeng Wang</strong>');
  
  // Format venue with bold abbreviation if it exists
  let venueHTML = '';
  if (pub.venue.abbreviation) {
    venueHTML = `${pub.venue.fullName} (<strong>${pub.venue.abbreviation}</strong>), ${pub.venue.year}`;
  } else {
    venueHTML = `${pub.venue.fullName}, ${pub.venue.year}`;
  }
  
  // Add award in red if it exists
  if (pub.award) {
    venueHTML += `<br><span style="color: red;">${pub.award}</span>`;
  }
  
  return `
    <tr data-tags="${pub.tags.join(' ')}">
      <td width="32%">
        <img class="thumbnail" src="${pub.image}" data-id="${carouselId}"></img>
      </td>
      <td style="padding:20px;width:75%;vertical-align:middle">
        <a href="${pub.paperLink}">
          <papertitle>${pub.title}</papertitle>
        </a>
        <br>
        ${authorsWithBold}
        <br>
        <em>${venueHTML}</em>
        <br>
        [<a href="${pub.paperLink}">paper</a>]
        ${pub.codeLink ? `[<a href="${pub.codeLink}">code</a>]` : ''}
        ${pub.blogLink ? `[<a href="${pub.blogLink}">blog</a>]` : ''}
        <br>
        <p></p>
        <p>${pub.description}</p>
      </td>
    </tr>
  `;
}

// Function to render all publications
function renderPublications() {
  const container = document.getElementById('publications-container');
  if (!container) return;

  // Sort publications by date (most recent first)
  const sortedPubs = [...publications].sort((a, b) => b.venue.year - a.venue.year);

  // Create carousel container if it doesn't exist
  if (!document.getElementById('carousel-container')) {
    const carouselContainer = document.createElement('div');
    carouselContainer.id = 'carousel-container';
    document.body.appendChild(carouselContainer);
  }

  // Render each publication
  container.innerHTML = sortedPubs.map(createPublicationHTML).join('');

  // Create carousels
  const carouselContainer = document.getElementById('carousel-container');
  carouselContainer.innerHTML = sortedPubs.map(pub => {
    const carouselId = pub.image.split('/').pop().split('.')[0] + '-carousel';
    return `
      <div id="${carouselId}" class="carousel">
        <div class="carousel-content">
          <div class="carousel-title"><a href="${pub.paperLink}" target="_blank">${pub.title}</a></div>
          <img class="img-center-carousel" src="${pub.image}" alt="">
          <div class="carousel-prompt">Click here to close</div>
        </div>
      </div>
    `;
  }).join('');

  // Initialize carousel functionality
  $(".carousel").on('click', function(e) {
    // Only close if clicking the background or the prompt
    if (e.target === this || e.target.classList.contains('carousel-prompt')) {
      $(this).hide();
      // Remove carousel-active class from the parent row
      $(this).closest('tr').removeClass('carousel-active');
    }
  });

  $(".thumbnail").on('click', function(e) {
    e.stopPropagation();  // Prevent click from bubbling up
    const carousel_id = $(this).data("id");
    const $carousel = $("#" + carousel_id);
    const $row = $(this).closest('tr');
    
    // Add carousel-active class to the row
    $row.addClass('carousel-active');
    
    // Show carousel
    $carousel.show();
  });
}

// Function to filter publications by tag
function filterPublications(tag) {
  const rows = document.querySelectorAll('#publications-container tr');
  rows.forEach(row => {
    if (tag === 'all' || row.dataset.tags.includes(tag)) {
      row.style.display = '';
    } else {
      row.style.display = 'none';
    }
  });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  renderPublications();
  
  // Set up click handlers for filter buttons
  document.querySelectorAll('button[data-tag]').forEach(button => {
    button.addEventListener('click', () => {
      const tag = button.dataset.tag;
      filterPublications(tag);
      
      // Update active button state
      document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.toggle('active', btn === button);
      });
    });
  });
}); 