<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/happy-panda-ops/xAI_Masterthesis_Pan">
    <img src="images/Otto-Friedrich-University_Bamberg_logo.png" alt="Logo" width="80" height="80" style="vertical-align: bottom;">
  </a>
  <a href="https://github.com/happy-panda-ops/xAI_Masterthesis_Pan">
    <img src="images/xaiLogo.png" alt="Logo" width="140" height="80" style="vertical-align: bottom;">
  </a>
  <a href="https://github.com/happy-panda-ops/xAI_Masterthesis_Pan">
    <img src="images/kdwt.png" alt="Logo" width="120" height="70" style="vertical-align: bottom;">
  </a>
  <a href="https://github.com/happy-panda-ops/xAI_Masterthesis_Pan">
    <img src="images/ddt.png" alt="Logo" width="160" height="80" style="vertical-align: bottom;">
  </a>
  </a>

<h3 align="center">AI-assisted wood knots detection
from historic timber structure
imaging</h3>

  <p align="center">
    This git repository contains the codes, training results and test code for Junquan Pan's master thesis at the Chair xAI of the University of Bamberg.
    <br />
    <a href="https://github.com/happy-panda-ops/xAI_Masterthesis_Pan"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/happy-panda-ops/xAI_Masterthesis_Pan/tree/main/Test_code">View Demo</a>
    ·
    <a href="https://github.com/happy-panda-ops/xAI_Masterthesis_Pan/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/happy-panda-ops/xAI_Masterthesis_Pan/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

<!-- Here's a blank template to get started: To avoid retyping too much info. Do a search and replace with your text editor for the following: `Holms_Pan`, `repo_name`, `JunquanPan`, `linkedin_username`, `email_client`, `email`, `project_title`, `project_description` -->

<!-- # Automated Wood Knot Detection for Historic Timber Structures -->

This project focuses on developing an **AI-powered system** for detecting wood knots on historic timber surfaces, contributing to the preservation and sustainability of invaluable heritage structures. Traditional manual methods for assessing timber integrity are often time-consuming, error-prone, and limited in challenging conditions. This project introduces a modern, automated solution leveraging **machine learning** and **deep learning** techniques.

## Key Features
1. **Two-Stage Detection Process:**
   - **Stage 1:** Timber surface segmentation using models like [Detectron2](https://github.com/facebookresearch/detectron2).
   - **Stage 2:** Knot detection with [YOLOv8](https://github.com/ultralytics/yolov8).
2. **Geometric Analysis:** Realistic measurements from mobile phone sensors are integrated to estimate knot dimensions.
3. **Dataset Integration:** Tests conducted on both labeled datasets and unseen collections ensure the system's reliability and adaptability.

## Goals
- **Accurate Documentation:** Provide detailed analysis of timber surfaces to assist heritage conservation.
- **Efficiency:** Reduce time and errors associated with traditional methods.
- **Accessibility:** Implement a mobile application for conservators to streamline inspection and analysis.

## Future Enhancements
Ongoing research aims to refine model accuracy and stability for diverse environmental conditions, ensuring a robust and reliable tool for the conservation community.

---

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* <a href="https://www.python.org/"><img src="https://www.python.org/static/community_logos/python-logo.png" alt="Python" width="100" height="30"/></a>
* <a href="https://jupyter.org/"><img src="https://jupyter.org/assets/homepage/main-logo.svg" alt="Jupyter Notebook" width="100" height="30"/></a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED
## Getting Started

The Train_code folder contains the training code and some of the training results based on detectron2 and YOLOv8.

The folder Test_code contains the compiled code for automatic detection of wood knots. The models used are currently set by default. More information about the required environment, packages and setup can be found in the "README" file in the corresponding folder. 

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Code Guide

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/happy-panda-ops/xAI_Masterthesis_Pan.git
   ```
3. Install related packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```
5. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin Holms_Pan/repo_name
   git remote -v # confirm the changes
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
 -->


<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
<!-- ## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/happy-panda-ops/xAI_Masterthesis_Pan/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ### Top contributors:

<a href="https://github.com/happy-panda-ops/xAI_Masterthesis_Pan/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Holms_Pan/repo_name" alt="contrib.rocks image" />
</a> -->



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@JunquanPan](https://twitter.com/JunquanPan) - junqan_pan@163.com

Project Link: [https://github.com/happy-panda-ops/xAI_Masterthesis_Pan](https://github.com/happy-panda-ops/xAI_Masterthesis_Pan)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS
## Acknowledgments

* []()
* []()
* []() -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Holms_Pan/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/happy-panda-ops/xAI_Masterthesis_Pan/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Holms_Pan/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/happy-panda-ops/xAI_Masterthesis_Pan/network/members
[stars-shield]: https://img.shields.io/github/stars/Holms_Pan/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/happy-panda-ops/xAI_Masterthesis_Pan/stargazers
[issues-shield]: https://img.shields.io/github/issues/Holms_Pan/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/happy-panda-ops/xAI_Masterthesis_Pan/issues
[license-shield]: https://img.shields.io/github/license/Holms_Pan/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/happy-panda-ops/xAI_Masterthesis_Pan/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
